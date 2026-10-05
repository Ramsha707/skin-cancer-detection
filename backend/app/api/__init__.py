"""API routers for the SkinFL backend."""

from fastapi import APIRouter

from app.api import agents, audit, detection, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(agents.router)
api_router.include_router(audit.router)
api_router.include_router(detection.router)

__all__ = ["api_router"]