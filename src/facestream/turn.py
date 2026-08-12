"""ICE server selection, including Cloudflare's TURN service.

WebRTC needs a relay when neither peer can be reached directly, which on most
cellular networks is the normal case. Cloudflare Realtime provides one, but only
issues short-lived credentials, so they have to be minted per session from a
long-lived key.
"""

import logging

import aiohttp

logger = logging.getLogger(__name__)

CREDENTIALS_URL = (
    "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}"
    "/credentials/generate-ice-servers"
)

DEFAULT_STUN = [{"urls": "stun:stun.l.google.com:19302"}]

# Comfortably longer than any session, and well inside what the API accepts.
CREDENTIAL_TTL_SECONDS = 86400

REQUEST_TIMEOUT_SECONDS = 10


def parse_ice_servers(data: dict) -> list[dict]:
    """Pull the ICE server list out of a Cloudflare credentials response.

    The API answers with a single ``iceServers`` object, while
    ``RTCPeerConnection`` expects a list of them.
    """
    ice_servers = data.get("iceServers")
    if not ice_servers:
        raise ValueError(f"No iceServers in the Cloudflare response: {data}")
    if isinstance(ice_servers, dict):
        return [ice_servers]
    if isinstance(ice_servers, list):
        return ice_servers
    raise ValueError(f"Unexpected iceServers type: {type(ice_servers).__name__}")


async def fetch_cloudflare_ice_servers(key_id: str, api_token: str) -> list[dict]:
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            CREDENTIALS_URL.format(key_id=key_id),
            headers={"Authorization": f"Bearer {api_token}"},
            json={"ttl": CREDENTIAL_TTL_SECONDS},
        ) as response:
            if response.status not in (200, 201):
                body = await response.text()
                raise RuntimeError(
                    f"Cloudflare TURN returned {response.status}: {body[:300]}"
                )
            data = await response.json()

    return parse_ice_servers(data)


async def get_ice_servers(key_id: str | None, api_token: str | None) -> list[dict]:
    """TURN when it is configured and reachable, STUN otherwise."""
    if not (key_id and api_token):
        logger.info("Using Google STUN")
        return DEFAULT_STUN

    try:
        ice_servers = await fetch_cloudflare_ice_servers(key_id, api_token)
    except Exception:
        # Most networks connect over STUN alone, so a TURN outage or a bad
        # token should degrade the session rather than end it.
        logger.exception("Cloudflare TURN unavailable, falling back to STUN")
        return DEFAULT_STUN

    logger.info("Using Cloudflare TURN servers")
    return ice_servers
