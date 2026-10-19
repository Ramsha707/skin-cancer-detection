"""API routers for the SkinFL backend."""

from fastapi import APIRouter

from app.api import agents, audit, detection, experiments, federated, models, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(agents.router)
api_router.include_router(audit.router)
api_router.include_router(detection.router)
api_router.include_router(experiments.router)
api_router.include_router(federated.router)
api_router.include_router(models.router)

__all__ = ["api_router"]