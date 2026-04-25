from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .models import RawDocument

_DEFAULT_CONTENT_TYPE = "text/plain; charset=utf-8"
_FETCH_TIMEOUT = httpx.Timeout(30.0)
_REMOTE_SCHEMES = {"http", "https"}


def _is_remote_source(source: str) -> bool:
    return urlsplit(source).scheme in _REMOTE_SCHEMES


def _guess_content_type(path: Path) -> str:
    guessed_content_type, _ = mimetypes.guess_type(path.name)
    if guessed_content_type is None:
        return "text/plain"
    return guessed_content_type


async def _read_local_document(path: Path) -> RawDocument:
    if not path.exists():
        msg = f"Local document does not exist: {path}"
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Local document is not a file: {path}"
        raise IsADirectoryError(msg)

    body = await asyncio.to_thread(path.read_text, encoding="utf-8")
    return RawDocument(source=str(path), body=body, content_type=_guess_content_type(path))


def _get_response_content_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", _DEFAULT_CONTENT_TYPE)


async def _fetch_remote_document(source: str, client: httpx.AsyncClient) -> RawDocument:
    response = await client.get(source)
    response.raise_for_status()
    return RawDocument(source=source, body=response.text, content_type=_get_response_content_type(response))


async def fetch_document(source: str, *, client: httpx.AsyncClient | None = None) -> RawDocument:
    """Fetch a raw document from a local path or remote HTTP source."""
    normalized_source = source.strip()
    if not normalized_source:
        msg = "Document source URL must not be empty."
        raise ValueError(msg)

    if not _is_remote_source(normalized_source):
        return await _read_local_document(Path(normalized_source))

    if client is not None:
        return await _fetch_remote_document(normalized_source, client)

    async with httpx.AsyncClient(follow_redirects=True, timeout=_FETCH_TIMEOUT) as short_lived_client:
        return await _fetch_remote_document(normalized_source, short_lived_client)
