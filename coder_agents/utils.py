
import contextvars
import os
import sqlite3
import time
from datetime import datetime

# Every Streamlit session gets its own thread_id, but the database, the generated
# dev.py and the log file used to be process-wide. Two people using the app at once
# would then share one streamlit.db: the second person's Reset node DROPs the table
# out from under the first person's run, and their generated code overwrites each
# other's dev.py. The session id below scopes all three to a per-session directory.
#
# A ContextVar rather than a module global because Streamlit runs every session in
# the same process; a global would be shared by all of them. Each session's script
# runs on its own thread, and LangGraph propagates the context into its node
# executor, so nodes see the id set by the session that invoked the graph.
_session_id = contextvars.ContextVar("coder_agents_session_id", default="default")

SESSIONS_ROOT = "sessions"


def set_session_id(session_id):
    """Bind the current thread's session id. Call once per Streamlit script run."""
    _session_id.set(str(session_id))


def get_session_id():
    return _session_id.get()


def session_dir():
    """Return (creating if needed) the working directory for the current session."""
    path = os.path.join(os.getcwd(), SESSIONS_ROOT, get_session_id())
    os.makedirs(path, exist_ok=True)
    return path


def db_path():
    return os.path.join(session_dir(), "programmer.db")


def log_path():
    return os.path.join(session_dir(), "log.txt")


def dev_py_path():
    return os.path.join(session_dir(), "dev.py")


# ---------------------------------------------------------------------------
# Logging
#
# Everything below writes to sessions/<session_id>/log.txt. The goal is that the
# file can be read top to bottom to follow a run: which node ran, in what order,
# how long it took, what went in and what came out, and why the router branched
# the way it did.
# ---------------------------------------------------------------------------

_BANNER_WIDTH = 76
# Start time of the node currently executing, so log_done can report its duration
_node_started = contextvars.ContextVar("coder_agents_node_start", default=None)


def _stamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(text, file_name=None):
    """
    Append text to this session's log file, one timestamp per line.

    Multi-line values are split so every line carries a stamp; blank lines stay
    blank so the sections still breathe.
    """
    stamp = _stamp()
    with open(file_name or log_path(), 'a', encoding='utf-8') as f:
        for line in str(text).split("\n"):
            f.write(f"[{stamp}] {line}\n" if line.strip() else "\n")


def log_banner(title, char="="):
    """Write a full-width section header."""
    filler = max(3, _BANNER_WIDTH - len(str(title)) - 5)
    log("")
    log(f"{char * 3} {title} {char * filler}")


def log_node(name, iterations=None):
    """Mark entry into a graph node and start its timer."""
    _node_started.set(time.perf_counter())
    label = str(name) if iterations is None else f"{name}  (iteration {iterations})"
    log_banner(label)


def log_done(summary=""):
    """Mark the end of a node, reporting how long it took."""
    started = _node_started.get()
    elapsed = f"{time.perf_counter() - started:.2f}s" if started else "?"
    log(f"  done in {elapsed}" + (f" - {summary}" if summary else ""))


def log_route(source, destination, reason=""):
    """Record a router decision, so the path through the graph is readable."""
    log(f"  ROUTE  {source} -> {destination}" + (f"   ({reason})" if reason else ""))


def truncate(text, limit=200):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def describe_message(message, limit=200):
    """
    Render one history entry as a single scannable line.

    LangChain message reprs run to hundreds of characters of metadata, which made
    the old log unreadable. This keeps the parts that matter: role, any tool calls
    with their arguments, and a truncated preview of the content.
    """
    # Plain ("role", "text") tuples used by the manager and programmer histories
    if isinstance(message, tuple):
        role = message[0] if message else "?"
        content = message[1] if len(message) > 1 else ""
        return f"{str(role):<9}| {truncate(str(content).replace(chr(10), ' / '), limit)}"

    role = getattr(message, "type", type(message).__name__)
    parts = []

    for call in getattr(message, "tool_calls", None) or []:
        args = ", ".join(f"{k}={truncate(v, 80)}" for k, v in (call.get("args") or {}).items())
        parts.append(f"CALL {call['name']}({args})")

    call_id = getattr(message, "tool_call_id", None)
    if call_id is not None:
        parts.append(f"RESULT for {call_id}")

    content = getattr(message, "content", "")
    if content:
        parts.append(truncate(str(content).replace("\n", " / "), limit))

    return f"{str(role):<9}| " + "  ".join(parts) if parts else f"{str(role):<9}| (empty)"


def log_messages(label, messages, limit=200):
    """Log a whole history as one indexed line per entry."""
    log(f"  {label} ({len(messages)}):")
    for index, message in enumerate(messages):
        log(f"    [{index}] {describe_message(message, limit)}")


def log_block(label, text):
    """Log a multi-line value in full, indented under a label."""
    log(f"  {label}:")
    for line in str(text).split("\n"):
        log(f"    {line}")


#Define a function to store a message in the database
def store_message(sender: str, programmer_output: str):
    conn = sqlite3.connect(db_path())
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # prefix is the programmer's description of the approach; the verifier reads it
    # as the "Code Explanation" half of its prompt, so it has to be persisted too.
    prefix = programmer_output.prefix
    imports = programmer_output.imports
    install = programmer_output.install
    code = programmer_output.code
    cursor.execute('''
    INSERT INTO programmer_messages (prefix, install, imports, code, timestamp) 
    VALUES (?, ?, ?, ?, ?)
    ''', (prefix, install, imports, code, timestamp))
    conn.commit()
    conn.close()


#Function to retrieve messages (for checking)
def get_last_message():
    conn = sqlite3.connect(db_path())
    # Row factory lets callers access columns by name instead of integer index
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM programmer_messages ORDER BY id DESC LIMIT 1')
    result = cursor.fetchone()
    conn.close()
    return result
