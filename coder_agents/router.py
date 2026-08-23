
from .code_interpreter import SUCCESS
from .utils import log_route
import time


def router(state,max_iterations,wait_time=0):
    """
    Conditional edge function that decides the next graph node based on the
    current sender and the content/tool_calls of the latest generation.

    Returns one of: 'continue', 'tools', 'end', 'programmer'
    """
    # This is the router
    generation = state["generation"]
    sender = state["sender"]
    iterations = state["iterations"]

    if sender == "manager":
        time.sleep(wait_time)
        # Hard stop if we've exceeded the iteration budget
        if iterations >= max_iterations:
            log_route("Manager", "END", f"iteration cap {iterations}/{max_iterations} reached")
            return 'end'
        # Manager wants to call a tool (save_py_file or ask_human)
        if generation.tool_calls:
            names = ", ".join(call["name"] for call in generation.tool_calls)
            log_route("Manager", "Tools", f"wants {names}")
            return 'tools'
        # Manager signals the workflow is complete
        if "DONE" in generation.content:
            log_route("Manager", "END", "manager said DONE")
            return 'end'
        else:
            # Manager has instructions for the programmer — continue the loop
            log_route("Manager", "Programmer", "has instructions")
            return 'continue'

    if sender == "programmer":
        time.sleep(wait_time)
        # An AIMessage always has a .tool_calls attribute (usually an empty list), so
        # test the list itself rather than hasattr. The delivered Code result is a
        # BaseModel and has no such attribute, which falls through to the interpreter.
        if getattr(generation, "tool_calls", None):
            names = ", ".join(call["name"] for call in generation.tool_calls)
            log_route("Programmer", "Tools", f"wants {names}")
            return 'tools'
        else:
            log_route("Programmer", "Code Interpreter", "code delivered")
            return "continue"

    if sender == "code_interpreter":
        # Route to Verifier on success, back to Programmer on any error
        if SUCCESS in generation:
            log_route("Code Interpreter", "Verifier", "app ran clean")
            return "continue"
        else:
            log_route("Code Interpreter", "Programmer", "app failed, needs a fix")
            return "programmer"

    if sender == "verifier":
        # Verifier always routes back to Manager with its feedback
        log_route("Verifier", "Manager", "feedback ready")
        time.sleep(wait_time)
        return "continue"