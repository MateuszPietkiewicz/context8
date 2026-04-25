from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

type Payload = dict[str, object]
type PayloadFactory = Callable[..., Payload]


@pytest.fixture
def build_document_chunk_payload() -> PayloadFactory:
    template: Payload = {
        "id": "chunk-001",
        "package": "context8",
        "version": "latest",
        "source_url": "https://example.com/docs/context8/models",
        "title": "Models",
        "text": "Validated chunk text.",
        "ordinal": 0,
        "metadata": {
            "source": "reference",
            "section": {"slug": "models", "priority": 1},
            "tags": ["pydantic", "contracts"],
        },
    }

    def build(**overrides: object) -> Payload:
        payload = deepcopy(template)
        payload.update(overrides)
        return payload

    return build


@pytest.fixture
def document_chunk_payload(build_document_chunk_payload: PayloadFactory) -> Payload:
    return build_document_chunk_payload()


@pytest.fixture
def build_search_candidate_payload() -> PayloadFactory:
    template: Payload = {
        "id": "chunk-001",
        "text": "Ranked search result.",
        "score": 0.97,
        "package": "context8",
        "version": "latest",
        "source_url": "https://example.com/docs/context8/search",
        "title": "Search",
        "ordinal": 0,
        "metadata": {
            "source": "search-index",
            "highlights": ["models", "validation"],
        },
    }

    def build(**overrides: object) -> Payload:
        payload = deepcopy(template)
        payload.update(overrides)
        return payload

    return build


@pytest.fixture
def search_candidate_payload(build_search_candidate_payload: PayloadFactory) -> Payload:
    return build_search_candidate_payload()


@pytest.fixture
def candidate_payloads(build_search_candidate_payload: PayloadFactory) -> list[Payload]:
    return [
        build_search_candidate_payload(),
        build_search_candidate_payload(
            id="chunk-002",
            score=0.84,
            ordinal=1,
            text="Another ranked result.",
            metadata={"source": "search-index", "highlights": ["rerank"]},
        ),
    ]


@pytest.fixture
def build_embed_response_payload() -> PayloadFactory:
    template: Payload = {
        "dimensions": 3,
        "vectors": [
            [0.11, 0.22, 0.33],
            [0.44, 0.55, 0.66],
        ],
    }

    def build(**overrides: object) -> Payload:
        payload = deepcopy(template)
        payload.update(overrides)
        return payload

    return build
