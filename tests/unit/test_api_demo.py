import re

import httpx

from fakes import FakeRetriever
from rag_engine.api.main import create_app
from rag_engine.core.config import Settings


def client(settings: Settings) -> httpx.AsyncClient:
    app = create_app(settings, probes=[], retriever=FakeRetriever([]))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_the_demo_page_is_served_as_html(settings: Settings) -> None:
    async with client(settings) as http:
        response = await http.get("/demo")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Hybrid RAG demo</title>" in response.text
    assert "/v1/query" in response.text


async def test_the_page_needs_no_api_key_because_the_key_is_typed_into_it(
    settings: Settings,
) -> None:
    from pydantic import SecretStr

    keyed = settings.model_copy(update={"api_keys": SecretStr("demo:0123456789abcdef-secret")})
    async with client(keyed) as http:
        page = await http.get("/demo")
        api = await http.post("/v1/search", json={"query": "x"})

    assert page.status_code == 200 and api.status_code == 401
    assert "0123456789abcdef-secret" not in page.text


async def test_only_the_pages_own_script_and_style_may_run(settings: Settings) -> None:
    async with client(settings) as http:
        response = await http.get("/demo")

    policy = response.headers["content-security-policy"]
    nonce = re.search(r"script-src \'nonce-([\w-]+)\'", policy)
    assert nonce, policy
    assert f'nonce="{nonce.group(1)}"' in response.text
    assert "default-src 'none'" in policy and "connect-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


async def test_every_visit_gets_a_new_nonce(settings: Settings) -> None:
    async with client(settings) as http:
        first = (await http.get("/demo")).headers["content-security-policy"]
        second = (await http.get("/demo")).headers["content-security-policy"]

    assert first != second


async def test_nothing_is_loaded_from_another_site_and_answers_never_become_markup(
    settings: Settings,
) -> None:
    async with client(settings) as http:
        page = (await http.get("/demo")).text

    assert not re.search(r"(src|href)=[\"\']https?:", page)
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page
    assert "eval(" not in page and "document.write" not in page


async def test_the_demo_is_not_listed_in_the_api_description(settings: Settings) -> None:
    async with client(settings) as http:
        schema = (await http.get("/openapi.json")).json()

    assert "/demo" not in schema["paths"]
