# File: main.py

import asyncio
import uuid # Not strictly needed here if Manus handles its own ID, but good for general utility
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, APIRouter, Request as FastAPIRequest
from pydantic import BaseModel # BaseModel is imported in app.models, but good to have if used directly here

# --- Import your custom modules ---
# Assuming Manus agent is in app.agent.manus
#from app.agent.manus import Manus
from app.agent.a2a_agent import Manus
# Assuming Pydantic models are in app.models
from app.models import AgentCard, AgentCapability, A2AMessageRequest, PromptRequest
# Assuming your logger is configured in app.logger
from app.logger import logger
# Assuming your config (if needed for APP_BASE_URL) might be in app.config
# from app.config import config as app_config # Example if you have a config object

app = FastAPI(title="Manus Agent API with A2A and Discovery")

# --- Global Variables ---
# Shared Manus agent instance
manus_agent: Optional[Manus] = None

# In-memory Agent Registry (for Discovery)
# In a production environment, use a persistent database (e.g., Redis, PostgreSQL)
AGENT_REGISTRY: Dict[str, AgentCard] = {}

# --- Configuration ---
# This is the publicly accessible base URL of this FastAPI application.
# It's crucial for constructing correct a2a_endpoints in agent cards.
# IMPORTANT: In production, get this from environment variables or a config file.
APP_BASE_URL = "http://localhost:8000"
# Example using environment variable:
# import os
# APP_BASE_URL = os.getenv("APP_PUBLIC_URL", "http://localhost:8000")

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Handles application startup events:
    1. Initializes the Manus agent.
    2. Registers the Manus agent with the in-memory discovery service.
    """
    global manus_agent
    logger.info(f"Application startup: Initializing Manus agent with base URL: {APP_BASE_URL}...")

    try:
        # Pass the base URL to the Manus agent so it can construct its public A2A endpoint
        manus_agent = await Manus.create(base_url=APP_BASE_URL)
        # The Manus.create() method now sets _manus_agent_id, _manus_public_name, _a2a_full_endpoint
        logger.info(f"Manus agent '{manus_agent._manus_public_name}' initialized with ID: {manus_agent._manus_agent_id}.") # type: ignore

        # Register this agent with the discovery service
        agent_card = await manus_agent.get_agent_card()
        AGENT_REGISTRY[agent_card.agent_id] = agent_card
        logger.info(f"Agent {agent_card.agent_id} registered with discovery service. Card: {agent_card.model_dump_json(indent=2)}")
    except Exception as e:
        logger.error(f"Critical error during Manus agent initialization: {e}", exc_info=True)
        # Depending on severity, you might want to prevent the app from starting
        # or run in a degraded mode. For now, it will continue but endpoints might fail.
        manus_agent = None # Ensure manus_agent is None if creation failed

@app.on_event("shutdown")
async def shutdown_event():
    """
    Handles application shutdown events:
    1. Unregisters the Manus agent from the discovery service.
    2. Cleans up Manus agent resources.
    """
    global manus_agent
    if manus_agent:
        agent_id = manus_agent._manus_agent_id # type: ignore
        logger.info(f"Application shutdown: Cleaning up Manus agent '{manus_agent._manus_public_name}' ({agent_id})...") # type: ignore

        # Unregister from discovery
        if agent_id in AGENT_REGISTRY:
            del AGENT_REGISTRY[agent_id]
            logger.info(f"Agent {agent_id} unregistered from discovery service.")

        try:
            await manus_agent.cleanup()
            logger.info(f"Manus agent {agent_id} cleanup complete.")
        except Exception as e:
            logger.error(f"Error during Manus agent cleanup for {agent_id}: {e}", exc_info=True)
    else:
        logger.info("Application shutdown: No Manus agent instance to clean up.")


# --- Agent Information Endpoint ---
@app.get("/.well-known/agent-card.json", response_model=AgentCard, tags=["Agent Info"])
async def get_this_agent_card():
    """
    Provides the agent card for this Manus agent instance.
    This is a common pattern for agents to expose their capabilities.
    """
    if not manus_agent:
        logger.warning("Attempted to get agent card, but agent is not initialized.")
        raise HTTPException(status_code=503, detail="Agent not initialized or unavailable.")
    return await manus_agent.get_agent_card()


# --- Direct Interaction Endpoint (Original /run) ---
@app.post("/run", response_model=Dict[str, Any], tags=["Direct Interaction"])
async def run_prompt_direct(request: PromptRequest):
    """
    Allows direct interaction with the Manus agent by submitting a prompt.
    """
    prompt = request.prompt.strip()
    if not prompt:
        logger.warning("Empty prompt received for /run endpoint.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if not manus_agent:
        logger.error("Manus agent not initialized. Cannot process /run request.")
        raise HTTPException(status_code=503, detail="Agent not available. Initialization might have failed.")

    agent_id = manus_agent._manus_agent_id # type: ignore
    logger.info(f"Processing direct prompt request for Manus ({agent_id}): '{prompt}'")
    try:
        # The `run` method in Manus should handle the prompt processing.
        result = await manus_agent.run(prompt=prompt) # Pass prompt as named argument
        logger.info(f"Direct prompt request processing completed by Manus ({agent_id}).")
        return {"status": "success", "message": "Prompt processed successfully via /run.", "data": result}
    except Exception as e:
        logger.error(f"Error during direct agent run for Manus ({agent_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing the prompt: {str(e)}")


# --- Agent Discovery Service Endpoints ---
discovery_router = APIRouter(prefix="/discovery", tags=["Agent Discovery"])

@discovery_router.post("/register", status_code=201, response_model=AgentCard)
async def register_agent_with_discovery(agent_card: AgentCard):
    """
    Allows other agents to register their agent cards with this discovery service.
    """
    if agent_card.agent_id in AGENT_REGISTRY:
        logger.info(f"Agent {agent_card.agent_id} re-registering. Updating card in registry.")
    else:
        logger.info(f"New agent {agent_card.agent_id} registering. Card: {agent_card.model_dump_json(indent=2)}")
    AGENT_REGISTRY[agent_card.agent_id] = agent_card
    return agent_card

@discovery_router.get("/agents", response_model=List[AgentCard])
async def list_all_registered_agents():
    """Lists all agent cards currently registered with the discovery service."""
    return list(AGENT_REGISTRY.values())

@discovery_router.get("/agents/{agent_id}", response_model=AgentCard)
async def get_specific_agent_card_from_discovery(agent_id: str):
    """Retrieves the agent card for a specific agent ID from the registry."""
    agent_card = AGENT_REGISTRY.get(agent_id)
    if not agent_card:
        logger.warning(f"Agent card for ID {agent_id} not found in discovery registry.")
        raise HTTPException(status_code=404, detail="Agent not found in registry.")
    return agent_card

@discovery_router.delete("/unregister/{agent_id}", status_code=200, response_model=Dict[str, str])
async def unregister_agent_from_discovery(agent_id: str):
    """Allows agents to unregister themselves from the discovery service."""
    if agent_id not in AGENT_REGISTRY:
        logger.warning(f"Attempt to unregister non-existent agent ID {agent_id}.")
        raise HTTPException(status_code=404, detail="Agent not found for unregistration.")

    removed_card = AGENT_REGISTRY.pop(agent_id)
    logger.info(f"Agent {agent_id} unregistered via discovery endpoint. Was: {removed_card.a2a_endpoint}")
    return {"message": "Agent unregistered successfully", "agent_id": agent_id}

app.include_router(discovery_router)


# --- A2A (Agent-to-Agent) Communication Endpoint for THIS Manus Agent ---
a2a_router = APIRouter(prefix="/a2a", tags=["Agent-to-Agent (A2A) Communication"])

@a2a_router.post("/{target_agent_id}/message", response_model=Dict[str, Any])
async def receive_a2a_message(target_agent_id: str, request: A2AMessageRequest):
    """
    Receives an A2A message intended for this specific Manus agent instance.
    The `target_agent_id` in the URL path MUST match this agent's ID.
    """
    global manus_agent
    if not manus_agent:
        logger.error("A2A message received, but local Manus agent is not initialized.")
        raise HTTPException(status_code=503, detail="Local Manus agent not initialized or unavailable.")

    local_agent_id = manus_agent._manus_agent_id # type: ignore
    if local_agent_id != target_agent_id:
        logger.warning(f"Received A2A message for agent {target_agent_id}, but this agent is {local_agent_id}.")
        raise HTTPException(status_code=400, detail=f"Message intended for agent {target_agent_id}, this is {local_agent_id}.")

    logger.info(f"Manus agent ({local_agent_id}) received A2A message from {request.sender_agent_id} of type '{request.message_type}'. Payload: {request.payload}")
    try:
        # Delegate message handling to the Manus agent instance
        response_payload = await manus_agent.handle_a2a_message(request)
        return response_payload
    except Exception as e:
        logger.error(f"Error handling A2A message for Manus ({local_agent_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing A2A message: {str(e)}")

app.include_router(a2a_router)


# --- Example: Endpoint for THIS agent to call ANOTHER agent (Demonstration) ---
@app.post("/delegate-task", response_model=Dict[str, Any], tags=["A2A Outgoing Example"])
async def delegate_task_to_another_agent(target_agent_id_query: str, task_prompt: str):
    """
    An example endpoint for this Manus agent to discover another agent
    (from its own registry) and send it an A2A message to execute a task.
    """
    if not manus_agent:
        logger.error("Cannot delegate task: Manus agent not initialized.")
        raise HTTPException(status_code=503, detail="Local Manus agent not initialized.")

    # 1. Discover the target agent from our registry
    target_agent_card = AGENT_REGISTRY.get(target_agent_id_query)
    if not target_agent_card:
        logger.warning(f"Cannot delegate task: Target agent ID '{target_agent_id_query}' not found in registry.")
        raise HTTPException(status_code=404, detail=f"Target agent {target_agent_id_query} not found in registry.")

    # 2. (Optional but good practice) Check if the target agent lists the required capability.
    #    For this example, we assume the target can handle "execute_prompt".
    # capability_to_execute = "execute_prompt"
    # has_capability = any(cap.name == capability_to_execute for cap in target_agent_card.capabilities)
    # if not has_capability:
    #     logger.warning(f"Target agent {target_agent_id_query} does not list '{capability_to_execute}' capability.")
    #     raise HTTPException(status_code=400, detail=f"Target agent {target_agent_id_query} does not have '{capability_to_execute}' capability.")

    # 3. Prepare and send the A2A message using Manus agent's internal method
    payload_for_other_agent = {
        "capability_name": "execute_prompt", # The capability we want the other agent to run
        "params": {"prompt": task_prompt}    # The parameters for that capability
    }
    message_type_to_send = "execute_capability"

    current_agent_id = manus_agent._manus_agent_id # type: ignore
    logger.info(f"Manus agent ({current_agent_id}) attempting to delegate task to {target_agent_id_query}: '{task_prompt}'")

    try:
        # The `send_a2a_message_to_another_agent` method is part of your Manus class
        response_from_other_agent = await manus_agent.send_a2a_message_to_another_agent(
            target_agent_card=target_agent_card,
            message_type=message_type_to_send,
            payload=payload_for_other_agent
        )
    except Exception as e: # Catch errors from the sending process itself
        logger.error(f"Error encountered while Manus ({current_agent_id}) was trying to send A2A message to {target_agent_id_query}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send A2A message: {str(e)}")


    if response_from_other_agent:
        logger.info(f"Successfully delegated task to {target_agent_id_query}. Response: {response_from_other_agent}")
        return {"status": "delegation_successful", "response_from_target_agent": response_from_other_agent}
    else:
        logger.warning(f"Delegation to {target_agent_id_query} sent, but no conclusive response received or failed to send.")
        # This could mean the request was sent but the other agent had an issue,
        # or network issues prevented getting a clear response.
        # The send_a2a_message_to_another_agent method in Manus should log specifics.
        raise HTTPException(status_code=502, detail=f"Failed to get a conclusive response from agent {target_agent_id_query} after sending task.")

# --- To run this application (example using uvicorn) ---
# Save this file as main.py
# In your terminal, navigate to the directory containing main.py and run:
# uvicorn main:app --reload --port 8000
#
# Ensure your `app` directory with `agent/manus.py`, `models.py`, etc., is in the same
# root directory or your PYTHONPATH is set up correctly.
