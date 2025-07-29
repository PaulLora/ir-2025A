from fastapi import APIRouter

router = APIRouter(
    prefix="/actuator",
    tags=["Actuator"],
    include_in_schema=False,
)


@router.get("/health")
async def health():
    return {"status": "UP"}
