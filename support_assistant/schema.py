from typing import List, Literal
from pydantic import BaseModel, Field

# FastAPI to Validate API request data
class AskRequest(BaseModel):
    """Request body for a POST/ask"""
    query: str

# FastAPI to validate API response data
class AskResponse(BaseModel):
    # Defines structure of the response returned by API
    """Structured output schema enforced on every answer."""
    # Answer - generated answer text
    answer: str
    # Intent - Whether it is related to policy_question or general_question
    intent : Literal["policy_question", "general_question"]
    # Sources - list of chunk/document IDs used to provide the answer. 
    # It use an empty string if the caller doesn't provide answer.
    sources: List[str] = Field(default_factory= list)
    # Confidence - Deterministic in mock mode, since there is no LLM output to score
    # 'ge' - greater than or equal to 
    # 'le' - less than or equal to
    confidence: float = Field(ge= 0.0, le= 1.0)
    