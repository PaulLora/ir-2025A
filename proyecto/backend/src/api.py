from typing import Any
from typing import Callable
from typing import cast
from typing import TypeVar

from fastapi import APIRouter
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
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

class API(FastAPI):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def add_router(self, router: APIRouter) -> None:
        for route in router.routes:
            route = cast(APIRoute, route)

            version = self.__route_version(route)

            route.path = f"/{version}/{route.path.removeprefix('/').removesuffix('/')}"

            API.__update_description(route)

        super().include_router(router)

    def __route_version(self, route) -> str:
        ver: int = cast(int, getattr(route.endpoint, "_route_version", None))
        if not ver:
            raise ValueError(f"Route {route.path} is missing version. Use @route_version decorator")
        return f"v{ver}"

    @staticmethod
    def __update_description(route: APIRoute) -> None:
        desc = f"""
### {route.path}
{route.description}
        """.strip()
        route.description = desc
