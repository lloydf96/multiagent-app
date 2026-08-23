
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, START
import yaml
from .code_interpreter import code_interpreter_agent
from .manager import manager_agent
from .prompts import get_prompts
from .programmer import programmer_agent
from .reset import reset_agent
from .router import router
from .state import Code, GraphState
from .verifier import verifier_agent
from .tool import save_py_file, ask_human, tool_agent, tavily_tool

import functools
import os

load_dotenv()

# Load model names from config.yaml at module level
_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(_config_path, 'r') as _f:
    _config = yaml.safe_load(_f)
_models = _config['models']

# Fallback iteration budget when neither the caller nor MAX_ITERATIONS supplies one
DEFAULT_MAX_ITERATIONS = 50



def create_workflow(max_iterations=DEFAULT_MAX_ITERATIONS):

    manager_prompt,verifier_prompt,programmer_prompt = get_prompts()

    # Define which tools each agent can call
    manager_tools = [save_py_file,ask_human]
    # tavily belongs to the Programmer: its system prompt tells it to search for
    # syntax questions and interpreter errors. It used to be listed here but bound
    # to no chain at all, so the search tool the prompt advertised did not exist.
    programmer_tools = [tavily_tool]
    tools = manager_tools + programmer_tools  # master list for the tool node

    # Manager LLM: tool-calling support (save file + ask human)
    manager_llm = ChatOpenAI(temperature=0, model=_models['manager'], model_kwargs={"service_tier":"flex"})
    manager_llm_with_tools = manager_llm.bind_tools(manager_tools)
    manager_chain = manager_prompt | manager_llm_with_tools

    # Verifier LLM: no tools needed
    verifier_llm = ChatOpenAI(temperature=0, model=_models['verifier'],model_kwargs={"service_tier":"flex"})
    verifier_chain = verifier_prompt | verifier_llm

    # Programmer LLM. with_structured_output cannot be combined with a search tool:
    # it pins tool_choice to the Code schema, so the model could never reach tavily.
    # Instead Code is offered as one tool among them and the model picks — tavily when
    # it needs to look something up, Code when it is ready to deliver.
    programmer_llm = ChatOpenAI(temperature=0, model=_models['programmer'],model_kwargs={"service_tier":"flex"})
    programmer_chain = programmer_prompt | programmer_llm.bind_tools(programmer_tools + [Code])
    # Fallback used once the search budget is spent, or if the model replies in prose
    programmer_forced_chain = programmer_prompt | programmer_llm.bind_tools([Code], tool_choice="Code")

    manager_node = functools.partial(manager_agent,chain = manager_chain)
    reset_node = functools.partial(reset_agent)
    programmer_node = functools.partial(programmer_agent,chain = programmer_chain,forced_chain = programmer_forced_chain)
    verifier_node = functools.partial(verifier_agent, chain = verifier_chain)
    code_interpreter_node = functools.partial(code_interpreter_agent)
    tool_node = functools.partial(tool_agent,tools = tools)
    router_edge = functools.partial(router,max_iterations = max_iterations,wait_time = 0)

    workflow = StateGraph(GraphState)

    workflow.add_node("Manager", manager_node)
    workflow.add_node("Reset",reset_node)
    workflow.add_node("Programmer", programmer_node)
    workflow.add_node("Verifier", verifier_node)
    workflow.add_node("Code Interpreter", code_interpreter_node)
    workflow.add_node("Tools", tool_node)


    workflow.add_conditional_edges(
        "Manager",
        router_edge,
        {"continue": "Programmer", 'tools' : 'Tools','end': END},
    )

    workflow.add_conditional_edges(
        "Programmer",
        router_edge,
        {"continue": "Code Interpreter", 'tools': 'Tools'},
    )

    workflow.add_conditional_edges(
        "Code Interpreter",
        router_edge,
        {"continue": "Verifier", "programmer" : "Programmer"},
    )

    workflow.add_conditional_edges(
        "Verifier",
        router_edge,
        {"continue": "Manager"},
    )

    workflow.add_conditional_edges(
        "Tools",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "manager": "Manager",
            # Reached whenever the Programmer calls tavily. Without this entry the
            # lambda returns 'programmer' and the lookup raises an unhandled KeyError.
            "programmer": "Programmer",
        }
    )

    # Entry point: always reset state before the Manager runs
    workflow.add_edge(START, "Reset")
    workflow.add_edge('Reset','Manager')
    # Run the graph up to the breakpoint
    memory = MemorySaver() 
    graph = workflow.compile(checkpointer=memory)
    return graph

