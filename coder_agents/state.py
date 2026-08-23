from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Data model
class Code(BaseModel):
    """Deliver the finished code solution. Call this when you are ready to write
    code, rather than replying with prose."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    install: str = Field(description="pip install all libraries necessary to run the code. The library list would be the ones which are imported to run the code.")



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """
    messages: List
    manager_messages: List
    programmer_messages : List
    # The user's pending reply to an ask_human call; None when nothing is pending
    human_feedback : Optional[str]
    generation: str
    iterations: int
    sender: str