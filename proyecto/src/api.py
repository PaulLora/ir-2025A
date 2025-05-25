from datetime import datetime
from typing import Any
from typing import Callable
from typing import cast
from typing import TypeVar
from typing import Union

from fastapi import APIRouter
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from pydantic import BaseModel


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


class RouteDeprecation:

    def __init__(self, sunset_header: str):
        self.sunset_header = sunset_header

    def __call__(self, _: Request, response: Response):
        response.headers["Sunset"] = self.sunset_header


def route_version(major: int) -> Callable[[CallableT], CallableT]:
    def decorator(func: CallableT) -> CallableT:
        setattr(func, "_route_version", major)
        return func

    return decorator


class RouteDetail(BaseModel):
    path: str
    version: str
    sunset_date: Union[str | None]


def route_deprecated(sunset: str) -> Callable[[CallableT], CallableT]:
    def decorator(func: CallableT) -> CallableT:
        setattr(func, "_route_deprecated", sunset)
        return func

    return decorator


def customize_openapi(app: FastAPI):
    if app.openapi_schema:
        return
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        tags=app.openapi_tags,
        routes=app.routes,
        contact=app.contact,
    )

    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if openapi_schema["paths"][path][method]["responses"].get("422"):
                openapi_schema["paths"][path][method]["responses"].pop("422")
    app.openapi_schema = openapi_schema


class API(FastAPI):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def add_router(self, router: APIRouter) -> None:
        for route in router.routes:
            route = cast(APIRoute, route)

            version = self.__route_version(route)
            sunset_date = self.__add_deprecation_details(route)

            route.path = f"/{version}/{route.path.removeprefix('/').removesuffix('/')}"

            route_detail = RouteDetail(path=route.path, version=version, sunset_date=sunset_date)

            API.__update_description(route, route_detail)

        super().include_router(router)

    def __route_version(self, route) -> str:
        ver: int = cast(int, getattr(route.endpoint, "_route_version", None))
        if not ver:
            raise ValueError(f"Route {route.path} is missing version. Use @route_version decorator")
        return f"v{ver}"

    def __add_deprecation_details(self, route) -> Union[str, None]:
        sunset: str = cast(str, getattr(route.endpoint, "_route_deprecated", None))
        if route.deprecated and not sunset:
            raise ValueError(
                f"Route {route.path} is marked as deprecated, but no sunset date provided. Use @route_deprecated decorator"
            )

        readable_sunset: Union[str, None] = None

        if sunset:
            d = datetime.strptime(sunset, "%Y-%m-%d").date()
            readable_sunset = f"{d:%B %d, %Y}"

            route.deprecated = True
            route.dependencies.append(Depends(RouteDeprecation(readable_sunset)))

        return readable_sunset

    @staticmethod
    def __update_description(route: APIRoute, rd: RouteDetail) -> None:
        desc = f"""
### {route.path}
{(rd.sunset_date and f"#### **Deprecation Date**: {rd.sunset_date}") or ""}
{route.description}
        """.strip()
        route.description = desc
