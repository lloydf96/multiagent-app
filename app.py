import streamlit as st
from coder_agents.workflow import create_workflow
from coder_agents.utils import log, log_banner, log_block, set_session_id, dev_py_path, log_path
import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

def llm_api(question,graph,config):
    """
    Simulate an LLM API call by echoing the user input.
    In production, replace this with an actual API call.
    """
    events = graph.invoke({
                "manager_messages": [("user", question)], 
                "messages" : [], 
                "iterations": 0,
                "sender" : "start",
                "generation" : "",
                "human_feedback" : None,
                "programmer_messages" : []
            }
        ,config
    )

    return events

# Set up the page configuration
st.set_page_config(page_title="ChatGPT-like Interface", layout="centered")

# Validate required environment variables before initializing any agents
_REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "Required for all LLM agents (Manager, Programmer, Verifier)",
    "TAVILY_API_KEY": "Required for the Verifier's web search tool",
    "E2B_API_KEY": "Required for the Code Interpreter's sandbox",
}
_missing = [k for k in _REQUIRED_ENV_VARS if not os.getenv(k)]
if _missing:
    for var in _missing:
        st.error(f"**{var}** is not set — {_REQUIRED_ENV_VARS[var]}. Add it to your `.env` file.")
    st.stop()

st.title("Code Writer")
# Maximum number of agent loop iterations before force-stopping
MAX_ITERATIONS = 50

# Each Streamlit session gets its own thread_id so concurrent users don't share state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Bind the session id for this script run before anything touches the database, the
# log or dev.py. Streamlit runs every session in one process, so this has to be re-set
# on each rerun; it scopes all three files to sessions/<session_id>/.
set_session_id(st.session_state.session_id)

# LangGraph execution config: thread_id keeps conversation memory; recursion_limit is 2× iterations
config = {"configurable": {"thread_id": st.session_state.session_id},"recursion_limit": MAX_ITERATIONS*2}

# Initialize graph
if 'graph' not in st.session_state:
    # Only this session's own files, so opening a second tab no longer wipes the
    # output another user is still working on
    for stale_path in (dev_py_path(), log_path()):
        if os.path.exists(stale_path):
            os.remove(stale_path)
    st.session_state['graph']  = create_workflow(MAX_ITERATIONS)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'start' not in st.session_state:
    st.session_state.start = True

if "download_button" not in st.session_state:
    st.session_state.download_button = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    graph = st.session_state.graph
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.session_state.turn = st.session_state.get("turn", 0) + 1
    log_banner(f"USER TURN {st.session_state.turn}", char="#")
    log_block("user said", prompt)

    turn_started = time.perf_counter()
    if st.session_state.start:
        # First user message: invoke the graph fresh from the START node
        log("  first call - running the graph from START")
        llm_api(prompt,graph,config)
        st.session_state.start = False
    else:
        # Subsequent messages: inject human feedback into the paused graph and resume
        pending = graph.get_state(config).next
        log(f"  resuming paused graph, pending node(s): {pending or '(none)'}")
        graph.update_state(config,{"human_feedback":prompt})
        graph.invoke(None,config)

    log_banner(f"TURN {st.session_state.turn} COMPLETE "
               f"({time.perf_counter() - turn_started:.1f}s)", char="#")

    # Determine the assistant's response from the final graph state
    values = graph.get_state(config).values
    generation = values['generation']
    tool_calls = getattr(generation, 'tool_calls', None) or []
    # A single manager turn can emit several tool calls — its prompt tells it to club
    # questions together and to follow save_py_file with ask_human — so search all of
    # them for the pending question instead of assuming it is the first one.
    question = next(
        (call['args']['question'] for call in tool_calls if call['name'] == 'ask_human'),
        None,
    )
    if question is not None:
        # Graph is waiting for the user — surface the question it wants to ask
        response = question
    elif tool_calls:
        # Unexpected tool call still pending; likely hit the iteration cap
        response = f'Internal state error due to limits on iterations. {values["iterations"]}'
    else:
        # Normal completion: return the plain text content from the manager LLM
        response = generation.content

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
            # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Show a download button if the agent has already written a dev.py output file
if os.path.exists(dev_py_path()):
        with open(dev_py_path(),'r',encoding='utf-8') as f:
            data = f.read()
            st.download_button("Download dev.py file",file_name = 'dev.py',data = data, icon = ":material/download:")
