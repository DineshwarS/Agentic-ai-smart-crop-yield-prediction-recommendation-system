import uuid
from fastapi import APIRouter

from ..schemas.schemas import AgentChatRequest, AgentChatResponse
from ..services.agent_service import agent_chat

router = APIRouter()


@router.post("/agent/chat", response_model=AgentChatResponse)
async def chat(req: AgentChatRequest):
    """
    Send a message to the Agentic AI assistant.
    The agent can call get_weather, predict_yield, recommend_crop, compare_crops tools.
    Requires GEMINI_API_KEY environment variable.
    """
    session_id = req.session_id or str(uuid.uuid4())
    result = await agent_chat(
        user_message=req.message,
        session_id=session_id,
        context=req.context,
    )
    return AgentChatResponse(
        reply=result["reply"],
        session_id=result["session_id"],
        tool_calls_made=result.get("tool_calls_made", []),
        structured_data=result.get("structured_data"),
    )
