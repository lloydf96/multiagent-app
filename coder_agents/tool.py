from .utils import log, log_done, log_node, get_last_message, dev_py_path, truncate
from langchain_core.messages import HumanMessage,AIMessage, ToolMessage
from langgraph.errors import NodeInterrupt
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

@tool
def save_py_file():
    """
    Saves the code stored by programmer in a database to a .py file. at the beginning of each user request the database is empty.
    """
    try:
        result = get_last_message()
        py_file = result['imports'] + "\n\n" + result['code']
        with open(dev_py_path(), "w",encoding="utf-8") as f:
            f.write(py_file)
        return "File Written"
    except Exception as e:
        return f"Data doesnt exist: {e}"
    

@tool
def ask_human(question: str):
    """
    Use this tool to ask user a question.
    Args:
        question: Question that you have for the user
    """
    # Never actually invoked: tool_agent intercepts this call, pauses the graph via
    # NodeInterrupt, and supplies the user's reply as the ToolMessage content.
    return None

# Description shown to the LLM so it knows when to invoke the Tavily search tool
TAVILY_DESCRIPTION = ('A search engine optimized for comprehensive, accurate,'
                     'and trusted results. Useful for when you need an answer to a programming question you have.'
                     'Input should be a search query')

# Restrict Tavily searches to these domains to get high-quality programming answers
TAVILY_DOMAIN = ['https://stackoverflow.com/','https://docs.streamlit.io/']

# Tavily search tool configured for advanced, text-only results from trusted dev sources
tavily_tool = TavilySearchResults(
    max_results=5,
    description=TAVILY_DESCRIPTION,
    include_domains=TAVILY_DOMAIN,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

def tool_agent(state,tools):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---TOOL WORKING---")
    log_node("TOOLS", state["iterations"])

    # State
    generation = state["generation"]
    tool_calls = generation.tool_calls
    log(f"  called by: {state['sender']}")

    # Build a lookup dict so we can dispatch by tool name
    tools_by_name = {t.name: t for t in tools}

    # Check for a pending ask_human BEFORE invoking anything. LangGraph has no
    # partial-progress checkpoint within a node: raising NodeInterrupt mid-loop
    # discards the observations gathered so far and re-runs the whole node on
    # resume, so any tool already invoked would run a second time. Interrupting
    # up front keeps the node all-or-nothing.
    if state.get('human_feedback') is None:
        for tool_call in tool_calls:
            if tool_call['name'] == 'ask_human':
                log(f"  PAUSED waiting for human: {tool_call['args'].get('question')}")
                raise NodeInterrupt("Waiting for human response")

    result = []
    for tool_call in tool_calls:
        log(f"  CALL {tool_call['name']}({truncate(tool_call.get('args'), 200)})")
        if tool_call['name'] == 'ask_human':
            # Human has replied: wrap the feedback as a ToolMessage and clear it
            observation = ToolMessage(content = state['human_feedback'],tool_call_id = tool_call['id'])
            state['human_feedback'] = None
        elif tool_call['name'] in tools_by_name:
            # Regular tool (e.g. save_py_file, tavily_tool): invoke and wrap the result.
            # Coerce to str: tavily returns a list of result dicts, and a ToolMessage
            # holding a list is sent as OpenAI content blocks, which are rejected with
            # "Missing required parameter: messages[N].content[0].type".
            tool_result = tools_by_name[tool_call['name']].invoke(tool_call["args"])
            if not isinstance(tool_result, str):
                tool_result = str(tool_result)
            observation = ToolMessage(content = tool_result,tool_call_id = tool_call['id'])
        else:
            # Hallucinated tool name: report it back so the LLM can correct itself,
            # rather than raising a KeyError that surfaces as a Streamlit traceback.
            log(f"  UNKNOWN TOOL: {tool_call['name']} (hallucinated)")
            observation = ToolMessage(
                content = f"Error: unknown tool '{tool_call['name']}'. Available tools: {', '.join(sorted(tools_by_name))}.",
                tool_call_id = tool_call['id'],
            )
        log(f"    -> {truncate(observation.content, 300)}")
        result.append(observation)

    log_done(f"{len(result)} tool result(s)")
    state['generation'] = result
    state['iterations'] += 1

    return state