from .utils import get_last_message, log, log_block, log_done, log_node

def verifier_agent(state, chain):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("\n\n---VERIFIER WORKING---")
    log_node("VERIFIER", state["iterations"])

    # State
    messages = state["messages"]
    code_solution = get_last_message()

    # Get solution components by name (via sqlite3.Row)
    prefix = code_solution['prefix']    # the programmer's description of the approach
    imports = code_solution['imports']
    code = code_solution['code']
    code = imports + "\n" + code  # combine into a single runnable block

    log(f"  reviewing {len(code.splitlines())} lines of code")
    log_block("approach under review", prefix)

    verifier_output = chain.invoke({"prefix": prefix, "code": code})

    log_block("feedback", verifier_output.content)
    log_done()

    messages += [
        (
            "verifier",
            (
                "assistant",
                f"{verifier_output}",
            ))
    ]

    state['iterations'] += 1
    state['generation'] = verifier_output
    state['messages'] = messages
    state['sender'] = 'verifier'

    return state