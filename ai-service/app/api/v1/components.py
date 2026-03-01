"""Component catalog API endpoints."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Any, Dict

from app.models.schemas.component_catalog import export_component_catalog as export_component_catalog_payload

router = APIRouter()


@router.get(
    "/components",
    tags=["Components"],
    summary="Get full component catalog",
    description="Returns the full centralized component catalog used across the system."
)
async def get_component_catalog() -> Dict[str, Any]:
    return export_component_catalog_payload()


@router.get(
    "/components/export",
    tags=["Components"],
    summary="Export component catalog as JSON",
    description="Downloads the full component catalog as a JSON file."
)
async def export_component_catalog() -> JSONResponse:
    response = JSONResponse(
        content=export_component_catalog_payload()
    )
    response.headers["Content-Disposition"] = 'attachment; filename="component_catalog.json"'
    return response
