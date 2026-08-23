
from langchain_core.messages import HumanMessage

from .utils import describe_message, log, log_block, log_done, log_messages, log_node

# How many recent standalone messages to backfill into the manager's context
MAX_INPUT_MESSAGES = 4


def _tool_call_ids(message):
    """Return the tool_call_ids requested by an AI message (empty list if none)."""
    return [call["id"] for call in getattr(message, "tool_calls", None) or []]


def _tool_call_groups(manager_messages):
    """
    Map each message index to the set of indices that must travel with it.

    An AI message carrying tool_calls and every ToolMessage answering one of its
    tool_call_ids form a single atomic group. They have to be kept or dropped
    together: the OpenAI API rejects an assistant message whose tool_call_ids are
    not each answered by a tool message, so splitting a group produces a 400.

    Grouping is by tool_call_id rather than by position, because a single manager
    turn can emit several tool calls (the prompt tells it to club questions
    together, and to call save_py_file alongside ask_human), which appends several
    consecutive ToolMessages to the history.
    """
    # tool_call_id -> index of the ToolMessage answering it
    responder_of = {}
    for i, message in enumerate(manager_messages):
        call_id = getattr(message, "tool_call_id", None)
        if call_id is not None:
            responder_of[call_id] = i

    groups = {}
    for i, message in enumerate(manager_messages):
        call_ids = _tool_call_ids(message)
        if not call_ids:
            continue
        group = {i}
        for call_id in call_ids:
            if call_id in responder_of:
                group.add(responder_of[call_id])
        for index in group:
            groups[index] = group
    return groups


def get_manager_input_messages(manager_messages):
    """
    Filters the full manager message history down to a trimmed list for LLM input.

    Always keeps:
      - The first message (original user request)
      - Every AI message containing tool calls, together with all of its tool responses

    Then backfills up to MAX_INPUT_MESSAGES recent standalone messages from the tail
    of the history to keep the context window manageable.
    """
    if not manager_messages:
        return []

    groups = _tool_call_groups(manager_messages)

    # Always keep the original user request plus every complete tool-call exchange
    selected = {0}
    for group in groups.values():
        selected |= group

    # Backfill recent standalone messages from the end, up to the budget
    remaining = MAX_INPUT_MESSAGES
    for i in range(len(manager_messages) - 1, 0, -1):
        if remaining <= 0:
            break
        if i in selected:
            continue
        # An orphaned tool response — one whose requesting AI message is no longer in
        # the history — can never be sent on its own, so it is not a backfill candidate.
        if getattr(manager_messages[i], "tool_call_id", None) is not None:
            continue
        selected.add(i)
        remaining -= 1

    manager_input_messages = [manager_messages[i] for i in sorted(selected)]
    # Only show the untrimmed history when it actually differs from what was sent
    dropped = len(manager_messages) - len(manager_input_messages)
    if dropped:
        log_messages(f"full history ({dropped} trimmed out)", manager_messages)
    log_messages("sent to LLM", manager_input_messages)
    return manager_input_messages



def manager_agent(state, chain):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("\n\n---MANAGER WORKING---")
    log_node("MANAGER", state["iterations"])

    # States
    messages = state["messages"]
    manager_messages = state["manager_messages"]
    sender = state["sender"]
    log(f"  entered from: {sender}")

    if sender =='verifier':
        # The verifier's output is an AIMessage from a different chain. Appending it
        # as-is would enter the manager's history with role "assistant", making the
        # critique read as something the manager itself said. Wrap it as a
        # HumanMessage so the manager sees it as incoming input.
        #
        # Build a new message rather than mutating state['generation'] in place: that
        # object is already stored in the previous checkpoint, and editing it would
        # rewrite what the history says the verifier produced.
        verifier_output = state['generation']
        manager_messages.append(
            HumanMessage(content="Feedback from the Verifier:\n" + verifier_output.content)
        )
        log_block("verifier feedback taken in", verifier_output.content)

    if sender == "manager":
        tool_output = state["generation"]
        manager_messages += tool_output
        log_messages("tool results taken in", tool_output)

    manager_input_messages = get_manager_input_messages(manager_messages)

    manager_output = chain.invoke(
            {"messages": manager_input_messages}
            )

    # Tool calls go on one line; free text is worth showing in full, so don't do both
    if manager_output.tool_calls:
        log(f"  output: {describe_message(manager_output, limit=600)}")
    if manager_output.content:
        log_block("instructions", manager_output.content)
    log_done()

    messages += [
        (
            "manager",
            (
                "assistant",
                f"{manager_output}",
            ))
    ]

    manager_messages.append(manager_output)

    state['iterations'] += 1
    state['generation'] = manager_output
    state['messages'] = messages
    state['manager_messages'] = manager_messages
    state['sender'] = 'manager'
    return state
