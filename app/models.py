# File: app/models.py

from pydantic import BaseModel
from typing import List, Dict, Optional, Any  # <<<<<<<< IMPORT Any from typing

# --- Pydantic Models for Agent Cards and A2A ---

class AgentCapability(BaseModel):
    name: str
    description: Optional[str] = None
    # You could add input_schema, output_schema for more detailed capability description

class AgentCard(BaseModel):
    agent_id: str
    agent_name: str
    description: Optional[str] = None
    capabilities: List[AgentCapability] = []
    a2a_endpoint: str # Full URL for this agent to receive A2A messages
    # If you have a metadata field like this, ensure it uses typing.Any:
    # metadata: Optional[Dict[str, Any]] = None # <<<<<<<< Use typing.Any here

class A2AMessageRequest(BaseModel):
    sender_agent_id: str
    message_type: str # e.g., "execute_capability", "query_status"
    payload: Dict[str, Any] # <<<<<<<< THIS IS A VERY LIKELY PLACE FOR THE ERROR. Change 'any' to 'Any'

# This model is used by your existing /run endpoint and potentially by A2A calls
class PromptRequest(BaseModel):
    prompt: str
