#!/usr/bin/env python3
"""
E2E encrypted client test — ML-KEM-768 + HKDF + ChaCha20-Poly1305.

Usage: CHUTES_API_KEY=... python scripts/test_e2e_client.py
"""

import base64
import gzip
import json
import os
import sys

import httpx
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from loguru import logger
from pqcrypto.kem.ml_kem_768 import generate_keypair, encrypt, decrypt

API_BASE = "https://api.chutes.dev"
CHUTE_ID = "edd0810e-51f8-5119-b480-09c950812833"
MODEL = "unsloth/Llama-3.2-1B-Instruct"
E2E_PATH = "/v1/chat/completions"

API_KEY = os.environ.get("CHUTES_API_KEY")
if not API_KEY:
    logger.error("CHUTES_API_KEY env var is required")
    sys.exit(1)

AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}
MLKEM_CT_SIZE = 1088
TAG_SIZE = 16


def derive_key(shared_secret: bytes, mlkem_ct: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=mlkem_ct[:16], info=info).derive(
        shared_secret
    )


def chacha_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    ct_tag = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
    return ct_tag[:-TAG_SIZE], ct_tag[-TAG_SIZE:]


def chacha_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> bytes:
    return ChaCha20Poly1305(key).decrypt(nonce, ciphertext + tag, None)


def discover_instances() -> dict:
    url = f"{API_BASE}/e2e/instances/{CHUTE_ID}"
    resp = httpx.get(url, headers=AUTH_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info(
        f"Discovered {len(data['instances'])} instance(s), nonces expire in {data['nonce_expires_in']}s"
    )
    return data


def build_e2e_blob(e2e_pubkey_b64: str, payload: dict, response_pk: bytes) -> bytes:
    e2e_pubkey = base64.b64decode(e2e_pubkey_b64)
    mlkem_ct, shared_secret = encrypt(e2e_pubkey)
    sym_key = derive_key(shared_secret, mlkem_ct, b"e2e-req-v1")
    logger.info("ML-KEM encapsulated, HKDF derived request key (info=e2e-req-v1)")

    payload["e2e_response_pk"] = base64.b64encode(response_pk).decode()
    compressed = gzip.compress(json.dumps(payload).encode())

    nonce = os.urandom(12)
    ciphertext, tag = chacha_encrypt(sym_key, nonce, compressed)
    blob = mlkem_ct + nonce + ciphertext + tag
    logger.info(
        f"E2E blob: {len(blob)} bytes (ct={len(mlkem_ct)}, nonce=12, encrypted={len(ciphertext)}, tag=16)"
    )
    return blob


def decrypt_response_blob(response_blob: bytes, response_sk: bytes) -> dict:
    mlkem_ct = response_blob[:MLKEM_CT_SIZE]
    nonce = response_blob[MLKEM_CT_SIZE : MLKEM_CT_SIZE + 12]
    ciphertext = response_blob[MLKEM_CT_SIZE + 12 : -TAG_SIZE]
    tag = response_blob[-TAG_SIZE:]

    shared_secret = decrypt(response_sk, mlkem_ct)
    sym_key = derive_key(shared_secret, mlkem_ct, b"e2e-resp-v1")
    logger.info("ML-KEM decapsulated, HKDF derived response key (info=e2e-resp-v1)")

    plaintext = gzip.decompress(chacha_decrypt(sym_key, nonce, ciphertext, tag))
    return json.loads(plaintext)


def decrypt_stream_init(response_sk: bytes, mlkem_ct: bytes) -> bytes:
    shared_secret = decrypt(response_sk, mlkem_ct)
    stream_key = derive_key(shared_secret, mlkem_ct, b"e2e-stream-v1")
    logger.info("Stream key exchange complete (info=e2e-stream-v1)")
    return stream_key


def decrypt_stream_chunk(enc_chunk_b64: str, stream_key: bytes) -> str:
    raw = base64.b64decode(enc_chunk_b64)
    return chacha_decrypt(stream_key, raw[:12], raw[12:-TAG_SIZE], raw[-TAG_SIZE:]).decode()


def invoke_headers(instance_id: str, nonce: str, stream: bool) -> dict:
    return {
        **AUTH_HEADERS,
        "X-Chute-Id": CHUTE_ID,
        "X-Instance-Id": instance_id,
        "X-E2E-Nonce": nonce,
        "X-E2E-Stream": str(stream).lower(),
        "X-E2E-Path": E2E_PATH,
        "Content-Type": "application/octet-stream",
    }


def test_non_streaming():
    logger.info("--- Non-streaming E2E chat completion ---")
    data = discover_instances()
    instance = data["instances"][0]

    response_pk, response_sk = generate_keypair()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
    }
    e2e_blob = build_e2e_blob(instance["e2e_pubkey"], payload, response_pk)

    resp = httpx.post(
        f"{API_BASE}/e2e/invoke",
        headers=invoke_headers(instance["instance_id"], instance["nonces"][0], stream=False),
        content=e2e_blob,
        timeout=60,
    )
    if resp.status_code != 200:
        logger.error(f"Request failed: {resp.status_code} {resp.text[:500]}")
        return False

    result = decrypt_response_blob(resp.content, response_sk)
    logger.success(f"Response:\n{json.dumps(result, indent=2)}")
    return True


def test_streaming():
    logger.info("--- Streaming E2E chat completion ---")
    data = discover_instances()
    instance = data["instances"][0]

    response_pk, response_sk = generate_keypair()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        "stream": True,
    }
    e2e_blob = build_e2e_blob(instance["e2e_pubkey"], payload, response_pk)

    stream_key = None
    chunks = 0
    content = ""

    with httpx.stream(
        "POST",
        f"{API_BASE}/e2e/invoke",
        headers=invoke_headers(instance["instance_id"], instance["nonces"][0], stream=True),
        content=e2e_blob,
        timeout=60,
    ) as resp:
        if resp.status_code != 200:
            logger.error(f"Stream failed: {resp.status_code} {resp.read()[:500]}")
            return False

        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "e2e_init" in event:
                mlkem_ct = base64.b64decode(event["e2e_init"])
                stream_key = decrypt_stream_init(response_sk, mlkem_ct)

            elif "e2e" in event:
                if stream_key is None:
                    logger.error("Received e2e chunk before e2e_init")
                    return False
                chunk_text = decrypt_stream_chunk(event["e2e"], stream_key)
                chunks += 1
                try:
                    chunk_data = json.loads(chunk_text)
                    for choice in chunk_data.get("choices", []):
                        c = choice.get("delta", {}).get("content", "")
                        if c:
                            content += c
                            print(c, end="", flush=True)
                except json.JSONDecodeError:
                    content += chunk_text
                    print(chunk_text, end="", flush=True)

            elif "usage" in event:
                logger.info(f"Usage: {event['usage']}")

            elif "e2e_error" in event:
                logger.error(f"E2E error: {event['e2e_error']}")
                return False

    print()
    logger.success(f"Streamed {chunks} chunks: '{content.strip()}'")
    return True


def main():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )

    logger.info(f"E2E Client Test — {API_BASE} — {MODEL}")

    results = {}
    for name, test in [("non_streaming", test_non_streaming), ("streaming", test_streaming)]:
        try:
            results[name] = test()
        except Exception:
            logger.exception(f"{name} test failed")
            results[name] = False

    logger.info(
        "Results: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items())
    )
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
