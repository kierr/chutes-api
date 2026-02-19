"""
Helper to send requests to miners.
"""

import hashlib
import time
import httpx
import orjson as json
from contextlib import asynccontextmanager
from typing import Any, Dict
from api.config import settings
from api.constants import MINER_HEADER, VALIDATOR_HEADER, NONCE_HEADER, SIGNATURE_HEADER


def get_signing_message(
    miner_ss58: str,
    nonce: str,
    payload_str: str | bytes | None,
    purpose: str | None = None,
    payload_hash: str | None = None,
) -> str:
    """
    Get the signing message for a request to a miner.
    """
    if payload_str:
        if isinstance(payload_str, str):
            payload_str = payload_str.encode()
        return f"{miner_ss58}:{settings.validator_ss58}:{nonce}:{hashlib.sha256(payload_str).hexdigest()}"
    elif purpose:
        return f"{miner_ss58}:{settings.validator_ss58}:{nonce}:{purpose}"
    elif payload_hash:
        return f"{miner_ss58}:{settings.validator_ss58}:{nonce}:{payload_hash}"
    else:
        raise ValueError("Either payload_str or purpose must be provided")


def sign_request(miner_ss58: str, payload: Dict[str, Any] | str | None = None, purpose: str = None):
    """
    Generate a signed request (for miner requests to validators).
    """
    nonce = str(int(time.time()))
    headers = {
        VALIDATOR_HEADER: settings.validator_ss58,
        MINER_HEADER: miner_ss58,
        NONCE_HEADER: nonce,
    }
    signature_string = None
    payload_string = None
    if payload is not None:
        if isinstance(payload, dict):
            headers["Content-Type"] = "application/json"
            payload_string = json.dumps(payload)
        else:
            if isinstance(payload, str):
                headers["Content-Type"] = "text/plain; charset=utf-8"
            else:
                headers["Content-Type"] = "application/octet-stream"
            payload_string = payload
        signature_string = get_signing_message(
            miner_ss58,
            nonce,
            payload_str=payload_string,
            purpose=None,
        )
    else:
        signature_string = get_signing_message(miner_ss58, nonce, payload_str=None, purpose=purpose)
    headers[SIGNATURE_HEADER] = settings.validator_keypair.sign(signature_string.encode()).hex()
    return headers, payload_string


class _HttpxResponseWrapper:
    """Wraps an httpx.Response to provide aiohttp-compatible attribute access.

    This enables gradual migration — callers can use either style:
      - response.status (aiohttp) or response.status_code (httpx)
      - await response.text() or response.text (httpx)
      - await response.json() or response.json() (httpx)
    """

    def __init__(self, response: httpx.Response):
        self._response = response

    @property
    def status(self) -> int:
        return self._response.status_code

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def headers(self):
        return self._response.headers

    @property
    def content(self):
        return self._response.content

    async def text(self) -> str:
        return self._response.text

    async def json(self):
        return self._response.json()

    async def read(self) -> bytes:
        return self._response.content

    def raise_for_status(self):
        self._response.raise_for_status()

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._response, name)


@asynccontextmanager
async def post(miner_ss58: str, url: str, payload: Dict[str, Any], instance=None, **kwargs):
    """
    Perform a post request to a miner.
    """
    headers = kwargs.pop("headers", {})
    new_headers, payload_data = sign_request(miner_ss58, payload=payload)
    headers.update(new_headers)
    timeout_val = kwargs.pop("timeout", 600)
    kwargs.pop("params", None)  # httpx uses params kwarg natively

    # Build per-request timeout for overriding pooled client defaults.
    req_timeout = httpx.Timeout(
        connect=10.0, read=float(timeout_val) if timeout_val else None, write=30.0, pool=10.0
    )

    if instance:
        from api.instance.connection import get_instance_client

        client, pooled = await get_instance_client(
            instance, timeout=int(timeout_val) if timeout_val else 600
        )
        try:
            response = await client.post(
                url, content=payload_data, headers=headers, timeout=req_timeout
            )
            yield _HttpxResponseWrapper(response)
        finally:
            if not pooled:
                try:
                    await client.aclose()
                except Exception:
                    pass
    else:
        async with httpx.AsyncClient(timeout=req_timeout) as client:
            response = await client.post(url, content=payload_data, headers=headers)
            yield _HttpxResponseWrapper(response)


@asynccontextmanager
async def patch(miner_ss58: str, url: str, payload: Dict[str, Any], instance=None, **kwargs):
    """
    Perform a patch request to a miner.
    """
    headers = kwargs.pop("headers", {})
    new_headers, payload_data = sign_request(miner_ss58, payload=payload)
    headers.update(new_headers)
    timeout_val = kwargs.pop("timeout", 600)

    req_timeout = httpx.Timeout(
        connect=10.0, read=float(timeout_val) if timeout_val else None, write=30.0, pool=10.0
    )

    if instance:
        from api.instance.connection import get_instance_client

        client, pooled = await get_instance_client(
            instance, timeout=int(timeout_val) if timeout_val else 600
        )
        try:
            response = await client.patch(
                url, content=payload_data, headers=headers, timeout=req_timeout
            )
            yield _HttpxResponseWrapper(response)
        finally:
            if not pooled:
                try:
                    await client.aclose()
                except Exception:
                    pass
    else:
        async with httpx.AsyncClient(timeout=req_timeout) as client:
            response = await client.patch(url, content=payload_data, headers=headers)
            yield _HttpxResponseWrapper(response)


@asynccontextmanager
async def get(miner_ss58: str, url: str, purpose: str, instance=None, **kwargs):
    """
    Perform a get request to a miner.
    """
    headers = kwargs.pop("headers", {})
    new_headers, _ = sign_request(miner_ss58, purpose=purpose)
    headers.update(new_headers)
    timeout_val = kwargs.pop("timeout", 600)
    params = kwargs.pop("params", None)

    req_timeout = httpx.Timeout(
        connect=10.0, read=float(timeout_val) if timeout_val else None, write=30.0, pool=10.0
    )

    if instance:
        from api.instance.connection import get_instance_client

        client, pooled = await get_instance_client(
            instance, timeout=int(timeout_val) if timeout_val else 600
        )
        try:
            response = await client.get(url, headers=headers, params=params, timeout=req_timeout)
            yield _HttpxResponseWrapper(response)
        finally:
            if not pooled:
                try:
                    await client.aclose()
                except Exception:
                    pass
    else:
        async with httpx.AsyncClient(timeout=req_timeout) as client:
            response = await client.get(url, headers=headers, params=params)
            yield _HttpxResponseWrapper(response)
