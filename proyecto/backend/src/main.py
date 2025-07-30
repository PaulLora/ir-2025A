from src import __version__
from src.api import API
from src.actuator import router as actuator
from src.ir import router as ir
from fastapi.middleware.cors import CORSMiddleware


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(actuator.router)
app.add_router(ir.router)
