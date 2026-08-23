from langchain_core.prompts import ChatPromptTemplate
import os

# Directory containing one .md file per agent prompt
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'system_prompts')

def _load_prompt(name: str) -> str:
    """Read and return the contents of coder_agents/system_prompts/<name>.md."""
    with open(os.path.join(_PROMPTS_DIR, f'{name}.md'), 'r') as f:
        return f.read()

def get_prompts():
    """
    Loads system prompts from individual .md files in coder_agents/system_prompts/
    and builds LangChain ChatPromptTemplate objects for each of the three agents.

    Returns:
        manager_prompt: Template for the manager agent (supports dynamic {messages})
        verifier_prompt: Template for the verifier agent (expects {prefix} and {code})
        programmer_prompt: Template for the programmer agent (supports dynamic {messages})
    """
    # Manager receives the ongoing conversation history via a placeholder
    manager_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _load_prompt('manager')),
            ("placeholder", "{messages}"),
        ]
    )

    # Verifier receives a fixed structure: code explanation + the code block
    verifier_prompt = ChatPromptTemplate(
        [
            ("system", _load_prompt('verifier')),
            ("user", "Code Explanation: {prefix}\nCode: {code}"),
        ]
    )

    # Programmer receives its own trimmed conversation history via a placeholder
    programmer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _load_prompt('programmer')),
            ("placeholder", "{messages}"),
        ]
    )

    return manager_prompt, verifier_prompt, programmer_prompt