from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uuid

def problem_response(status: int, title: str, detail: Optional[str], type_: str = "about:blank", instance: Optional[str] = None):
    body = {
        "type": type_,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance or f"urn:uuid:{uuid.uuid4()}",
    }
    return JSONResponse(status_code=status, content=body, media_type="application/problem+json")

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return problem_response(status=exc.status_code, title=exc.detail or "HTTP Error", detail=None)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return problem_response(status=422, title="Unprocessable Entity", detail=str(exc))
