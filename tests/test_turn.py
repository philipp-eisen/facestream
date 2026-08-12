"""ICE server selection against a stub Cloudflare endpoint."""

import pytest
from aiohttp import web

from facestream import turn

CLOUDFLARE_RESPONSE = {
    "iceServers": {
        "urls": [
            "stun:stun.cloudflare.com:3478",
            "turn:turn.cloudflare.com:3478?transport=udp",
            "turns:turn.cloudflare.com:5349?transport=tcp",
        ],
        "username": "abc123",
        "credential": "s3cret",
    }
}


def test_wraps_a_single_object_in_a_list():
    assert turn.parse_ice_servers(CLOUDFLARE_RESPONSE) == [
        CLOUDFLARE_RESPONSE["iceServers"]
    ]


def test_passes_a_list_through():
    assert turn.parse_ice_servers({"iceServers": [{"urls": "turn:x"}]}) == [
        {"urls": "turn:x"}
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"iceServers": None},
        {"errors": [{"code": 1004, "message": "nope"}]},
        {"iceServers": "turn:x"},
    ],
    ids=["empty", "null", "error body", "string"],
)
def test_rejects_unusable_responses(payload):
    with pytest.raises(ValueError):
        turn.parse_ice_servers(payload)


@pytest.fixture
async def cloudflare(aiohttp_server, monkeypatch):
    """A stub of Cloudflare's credentials endpoint.

    The key id selects the behaviour, so a test can ask for a rejected token or
    a gateway error.
    """
    requests = []

    async def handler(request):
        body = await request.json()
        requests.append(
            {
                "path": request.path,
                "auth": request.headers.get("Authorization"),
                "ttl": body.get("ttl"),
            }
        )
        behaviour = request.match_info["key"]
        if behaviour == "good":
            return web.json_response(CLOUDFLARE_RESPONSE, status=201)
        if behaviour == "unauthorized":
            return web.json_response({"errors": [{"code": 1000}]}, status=401)
        if behaviour == "gateway-error":
            return web.Response(text="<html>bad gateway</html>", status=502)
        if behaviour == "empty":
            return web.json_response({}, status=201)
        return web.Response(status=404)

    app = web.Application()
    app.router.add_post("/v1/turn/keys/{key}/credentials/generate-ice-servers", handler)
    server = await aiohttp_server(app)

    monkeypatch.setattr(
        turn,
        "CREDENTIALS_URL",
        f"http://{server.host}:{server.port}"
        "/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
    )
    return requests


@pytest.mark.asyncio
async def test_fetches_credentials(cloudflare):
    servers = await turn.fetch_cloudflare_ice_servers("good", "token123")

    assert servers == [CLOUDFLARE_RESPONSE["iceServers"]]
    assert cloudflare[-1]["auth"] == "Bearer token123"
    assert cloudflare[-1]["ttl"] == turn.CREDENTIAL_TTL_SECONDS
    assert cloudflare[-1]["path"].endswith("/credentials/generate-ice-servers")


@pytest.mark.asyncio
async def test_uses_turn_when_it_works(cloudflare):
    assert await turn.get_ice_servers("good", "token123") == [
        CLOUDFLARE_RESPONSE["iceServers"]
    ]


@pytest.mark.parametrize("behaviour", ["unauthorized", "gateway-error", "empty"])
@pytest.mark.asyncio
async def test_falls_back_to_stun_instead_of_failing(cloudflare, behaviour):
    """A TURN problem should degrade the session, not end it."""
    assert await turn.get_ice_servers(behaviour, "token123") == turn.DEFAULT_STUN


@pytest.mark.parametrize(
    "key_id,api_token",
    [(None, None), ("key", None), (None, "token"), ("", "")],
    ids=["neither", "key only", "token only", "blank"],
)
@pytest.mark.asyncio
async def test_uses_stun_without_complete_credentials(key_id, api_token):
    assert await turn.get_ice_servers(key_id, api_token) == turn.DEFAULT_STUN


@pytest.mark.asyncio
async def test_falls_back_when_cloudflare_is_unreachable(monkeypatch):
    monkeypatch.setattr(
        turn,
        "CREDENTIALS_URL",
        "http://127.0.0.1:9/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
    )
    assert await turn.get_ice_servers("key", "token") == turn.DEFAULT_STUN
