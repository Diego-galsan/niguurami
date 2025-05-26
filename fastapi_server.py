from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

#from app.agent.manus import Manus
from app.agent.a2a_agent import Manus
from app.logger import logger

app = FastAPI(title="Manus Agent API")

# Shared Manus agent instance
manus_agent = None


class PromptRequest(BaseModel):
    prompt: str


@app.on_event("startup")
async def startup_event():
    global manus_agent
    logger.info("Starting up and initializing Manus agent...")
    manus_agent = await Manus.create()
    logger.info("Manus agent initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    global manus_agent
    if manus_agent:
        logger.info("Cleaning up Manus agent...")
        await manus_agent.cleanup()
        logger.info("Cleanup complete.")


@app.post("/run")
async def run_prompt(request: PromptRequest):
    prompt = request.prompt.strip()
    if not prompt:
        logger.warning("Empty prompt received.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    logger.info("Processing prompt request...")
    try:
        await manus_agent.run(prompt)
        logger.info("Request processing completed.")
        return {"status": "success", "message": "Prompt processed successfully."}
    except Exception as e:
        logger.error(f"Error during agent run: {e}")
        raise HTTPException(status_code=500, detail="Error processing the prompt.")
