from fastapi import APIRouter

from src.api import route_version

router = APIRouter()

@router.get("/test")
@route_version(major=1)
async def test():
    return {"status": "UP"}
