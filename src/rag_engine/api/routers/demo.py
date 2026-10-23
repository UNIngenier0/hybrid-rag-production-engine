"""The demo page. It is only HTML: the API key, if there is one, is typed into it by the user and
sent to /v1/query, so the page itself needs no key."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from rag_engine.api.demo_page import render_demo

router = APIRouter(tags=["demo"])


@router.get("/demo", include_in_schema=False)
async def demo() -> HTMLResponse:
    page, policy = render_demo()
    return HTMLResponse(
        page,
        headers={
            "Content-Security-Policy": policy,
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )
