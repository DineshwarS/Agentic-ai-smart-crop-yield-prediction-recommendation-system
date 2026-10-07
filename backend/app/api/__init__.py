"""API package init — export all routers."""

from . import (
    agent_router,
    auth_router,
    performance_router,
    predictions_router,
    recommendation_router,
    weather_router,
    yield_router,
)

__all__ = [
    "agent_router",
    "auth_router",
    "performance_router",
    "predictions_router",
    "recommendation_router",
    "weather_router",
    "yield_router",
]
