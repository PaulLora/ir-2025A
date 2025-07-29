from src import __version__
from src.api import API
from src.actuator import router as actuator
from src.ir import router as ir


app = API(
    openapi_url="/openapi.json",
    title="Proyecto IR - 2025A",
    version=__version__.__version__,
    responses={
        401: {"description": "Unauthorized - Missing or Invalid Token"},
        403: {"description": "Forbidden - Insufficient Permissions"},
        404: {"description": "Not Found - Resource Not Found"},
        400: {"description": "Bad Request - Validation Errors"},
    },
    docs_url=None,
    redoc_url="/docs",
)


app.include_router(actuator.router)
app.add_router(ir.router)
