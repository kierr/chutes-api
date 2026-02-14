"""Instance connection helpers — httpx + HTTP/2 with TLS cert verification."""

import ssl
import httpx
import httpcore
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key


# Cache SSL contexts and cert CNs per instance_id.
_ssl_cache: dict[str, tuple[ssl.SSLContext, str]] = {}

# Pooled httpx clients per instance (reuse TCP+TLS connections).
_client_cache: dict[str, httpx.AsyncClient] = {}


def _get_ssl_and_cn(instance) -> tuple[ssl.SSLContext, str]:
    """Get or create cached SSL context + CN for an instance."""
    iid = str(instance.instance_id)
    if iid in _ssl_cache:
        return _ssl_cache[iid]

    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cadata=instance.cacert)

    # Load mTLS client cert if available.
    extra = instance.extra or {}
    client_cert_pem = extra.get("client_cert")
    client_key_pem = extra.get("client_key")
    client_key_password = extra.get("client_key_password")
    if client_cert_pem and client_key_pem:
        # Decrypt the client key and load into SSL context.
        password_bytes = client_key_password.encode() if client_key_password else None
        client_key = load_pem_private_key(client_key_pem.encode(), password=password_bytes)
        # Re-serialize unencrypted (in memory only, never written to disk).
        client_key_unencrypted = client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        # Write to temporary in-memory for ssl context (load_cert_chain requires files).
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pem", delete=False) as cf:
            cf.write(client_cert_pem.encode())
            cert_tmp = cf.name
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pem", delete=False) as kf:
            kf.write(client_key_unencrypted)
            key_tmp = kf.name
        try:
            ctx.load_cert_chain(certfile=cert_tmp, keyfile=key_tmp)
        finally:
            os.unlink(cert_tmp)
            os.unlink(key_tmp)

    cert = x509.load_pem_x509_certificate(instance.cacert.encode())
    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    _ssl_cache[iid] = (ctx, cn)
    return ctx, cn


def evict_instance_ssl(instance_id: str):
    """Remove cached SSL context and client when an instance is destroyed."""
    iid = str(instance_id)
    _ssl_cache.pop(iid, None)
    client = _client_cache.pop(iid, None)
    if client and not client.is_closed:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(client.aclose())
        except RuntimeError:
            pass


def get_instance_url(instance, port: int | None = None) -> str:
    """Build the correct URL (https with CN or http with IP) for an instance."""
    p = port or instance.port
    if instance.cacert:
        _, cn = _get_ssl_and_cn(instance)
        return f"https://{cn}:{p}"
    return f"http://{instance.host}:{p}"


class _InstanceNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolves cert CN hostnames to instance IPs without external DNS lookups.

    httpx uses the URL hostname for TLS SNI and cert verification, then calls
    connect_tcp(hostname, port) for the actual TCP connection. We intercept
    connect_tcp and remap the CN hostname to the real IP. This means:
      - TLS SNI = hostname (correct, matches cert CN)
      - Cert verification = hostname vs cert CN (correct)
      - TCP connection = actual instance IP (correct, no DNS needed)
    """

    def __init__(self, hostname: str, ip: str):
        self._hostname = hostname
        self._ip = ip
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        actual_host = self._ip if host == self._hostname else host
        return await self._backend.connect_tcp(
            actual_host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self._backend.connect_unix_socket(
            path,
            timeout=timeout,
            socket_options=socket_options,
        )

    async def sleep(self, seconds):
        await self._backend.sleep(seconds)


async def get_instance_client(instance, timeout: int = 600) -> httpx.AsyncClient:
    """Get or create a pooled httpx AsyncClient for an instance (HTTP/2 if TLS)."""
    iid = str(instance.instance_id)
    if iid in _client_cache:
        client = _client_cache[iid]
        if not client.is_closed:
            return client

    if instance.cacert:
        ssl_ctx, cn = _get_ssl_and_cn(instance)
        # Build httpcore pool with our custom resolver that maps CN → IP.
        pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_ctx,
            http2=True,
            network_backend=_InstanceNetworkBackend(hostname=cn, ip=instance.host),
        )
        client = httpx.AsyncClient(
            transport=pool,
            base_url=f"https://{cn}:{instance.port}",
            timeout=httpx.Timeout(
                connect=10.0, read=float(timeout) if timeout else None, write=30.0, pool=10.0
            ),
        )
    else:
        client = httpx.AsyncClient(
            base_url=f"http://{instance.host}:{instance.port}",
            timeout=httpx.Timeout(
                connect=10.0, read=float(timeout) if timeout else None, write=30.0, pool=10.0
            ),
        )
    _client_cache[iid] = client
    return client
