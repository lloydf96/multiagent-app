from .state import Code
from .utils import log, log_block, log_done, log_messages, log_node, store_message, truncate

# How many tavily round-trips the programmer may make before it must deliver code.
# Without a cap the Programmer -> Tools -> Programmer loop has no exit condition of
# its own and would only stop at the graph's recursion_limit, which surfaces as a
# GraphRecursionError traceback in the browser.
MAX_SEARCH_ROUNDS = 3

# Keep history short enough for the context window: the first message (original task)
# plus this many recent entries.
MAX_HISTORY_TAIL = 5


def _search_rounds(programmer_messages):
    """Count tool results gathered since the last plain-text turn."""
    rounds = 0
    for message in reversed(programmer_messages):
        # Plain ("user"/"assistant", text) tuples mark the start of the current turn
        if isinstance(message, tuple):
            break
        if getattr(message, "tool_call_id", None) is not None:
            rounds += 1
    return rounds


def _drop_orphan_tool_messages(messages):
    """
    Drop tool results whose requesting AI message is no longer present.

    Trimming the history can cut between an AI message carrying tool_calls and the
    ToolMessages answering them. OpenAI rejects a tool result that answers no visible
    call with a 400, so the orphaned half has to go rather than being sent alone.
    """
    answered = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            answered.add(call["id"])
    return [
        message for message in messages
        if getattr(message, "tool_call_id", None) is None
        or message.tool_call_id in answered
    ]


def programmer_agent(state, chain, forced_chain):
    """
    Write (or correct) the code solution.

    The programmer picks between two tools: the tavily search tool when it needs to
    look something up, and the Code schema when it is ready to deliver. A search call
    routes out to the Tools node and comes back here with the results; a Code call is
    parsed into the Code model and handed on to the Code Interpreter.

    Args:
        state (dict): The current graph state
        chain: prompt | llm bound with [tavily, Code], the model's choice
        forced_chain: prompt | llm bound with [Code] and tool_choice="Code"

    Returns:
        state (dict): Updated state
    """

    print("\n\n---PROGRAMMER WORKING---")
    log_node("PROGRAMMER", state["iterations"])

    # State
    messages = state["messages"]
    programmer_messages = state["programmer_messages"]
    sender = state["sender"]
    log(f"  entered from: {sender}")

    if sender == "programmer":
        # Returning from the Tools node: state['generation'] holds the ToolMessages
        # carrying the search results. Append them and let the model try again.
        programmer_messages = programmer_messages + list(state["generation"])
        log_messages("search results taken in", list(state["generation"]))
    else:
        if len(programmer_messages) > MAX_HISTORY_TAIL + 1:
            programmer_messages = (
                [programmer_messages[0]] + programmer_messages[-MAX_HISTORY_TAIL:]
            )
        # Build the next user message depending on who triggered this node
        if sender == "manager":
            question = ("Request from Manager: " + state["generation"].content
                        + " Use the search tool in case you need answers on some syntax "
                        "specific questions you have or if you face any errors/bugs in "
                        "code from code_interpreter.")
        else:  # code_interpreter
            question = ("Please correct the following error found in the code you just "
                        "generated :\n" + state["generation"])
        programmer_messages = programmer_messages + [("user", question)]
        log_block("task", question)

    before = len(programmer_messages)
    programmer_messages = _drop_orphan_tool_messages(programmer_messages)
    if len(programmer_messages) != before:
        log(f"  dropped {before - len(programmer_messages)} orphaned tool result(s)")

    rounds = _search_rounds(programmer_messages)
    log_messages("sent to LLM", programmer_messages)

    if rounds >= MAX_SEARCH_ROUNDS:
        # Search budget spent — force the model to produce code with what it has
        log(f"  search budget spent ({rounds}/{MAX_SEARCH_ROUNDS}); forcing Code output")
        programmer_output_message = forced_chain.invoke({"messages": programmer_messages})
    else:
        log(f"  searches used so far: {rounds}/{MAX_SEARCH_ROUNDS}")
        programmer_output_message = chain.invoke({"messages": programmer_messages})

    tool_calls = programmer_output_message.tool_calls or []
    code_call = next((call for call in tool_calls if call["name"] == "Code"), None)
    search_calls = [call for call in tool_calls if call["name"] != "Code"]

    if code_call is None and search_calls:
        # The model wants to search first. Hand off to the Tools node; the router
        # sees the tool_calls on this AIMessage and the Tools node routes back here
        # on 'programmer'. The AIMessage must go into the history so the ToolMessages
        # that answer it are not orphaned.
        for call in search_calls:
            log(f"  SEARCH {call['name']}({truncate(call.get('args'), 200)})")
        log_done("searching, will return to Programmer")
        state["generation"] = programmer_output_message
        state["sender"] = "programmer"
        state["programmer_messages"] = programmer_messages + [programmer_output_message]
        state["iterations"] += 1
        return state

    if code_call is None:
        # Model answered in prose instead of calling a tool; make it commit to Code
        log("  no tool call returned (replied in prose); forcing Code output")
        programmer_output_message = forced_chain.invoke({"messages": programmer_messages})
        code_call = next(
            (call for call in programmer_output_message.tool_calls or []
             if call["name"] == "Code"),
            None,
        )
        if code_call is None:
            raise RuntimeError("Programmer did not return a Code tool call even when forced")

    # Build the Code model field by field: a missing key would make Code(**args) raise
    # a ValidationError that surfaces as a Streamlit traceback.
    args = code_call["args"]
    programmer_output = Code(
        prefix=args.get("prefix", ""),
        imports=args.get("imports", ""),
        code=args.get("code", ""),
        install=args.get("install", ""),
    )

    programmer_output_str = f"prefix: {programmer_output.prefix}\ninstall: {programmer_output.install}\nimports: {programmer_output.imports}\ncode: {programmer_output.code}"

    log_block("approach", programmer_output.prefix)
    log_block("install", programmer_output.install)
    log_block("code", programmer_output.imports + "\n" + programmer_output.code)

    store_message('programmer', programmer_output)
    log_done(f"stored {len(programmer_output.code.splitlines())} lines of code")

    programmer_messages = programmer_messages + [("assistant", programmer_output_str)]
    messages += [
        (
            "programmer",
            (
                "assistant",
                f"{programmer_output}",
            ))
    ]

    state['iterations'] += 1
    state['generation'] = programmer_output
    state['messages'] = messages
    state['sender'] = 'programmer'
    state['programmer_messages'] = programmer_messages

    return state
