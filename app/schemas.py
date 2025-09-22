from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    """
    Schema for a user's chat message.
    """
    question: str
    messages: Optional[List[Dict[str, Any]]] = None
    session_id:  Optional[str] = None
    selected_pdf:str

class ChatResponse(BaseModel):
    """
    Schema for an assistant's response.
    """
    answer: str

class Message(BaseModel):
    """
    Schema for a single chat message in the history.
    """
    role: str
    content: str