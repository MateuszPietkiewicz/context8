from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from context8.models import (
    DocumentChunk,
    EmbedRequest,
    EmbedResponse,
    IngestReport,
    RerankRequest,
    RerankResponse,
    SearchCandidate,
    SearchResponse,
)

type PayloadFactory = Callable[..., dict[str, Any]]

pytestmark = pytest.mark.unit


@pytest.mark.contract
class TestDocumentChunk:
    def test_builds_valid_chunk_with_nested_metadata(self, document_chunk_payload: dict[str, Any]) -> None:
        chunk = DocumentChunk.model_validate(document_chunk_payload)

        assert chunk.id == "chunk-001"
        assert str(chunk.source_url) == "https://example.com/docs/context8/models"
        assert chunk.metadata["section"] == {"slug": "models", "priority": 1}
        assert chunk.metadata["tags"] == ["pydantic", "contracts"]

    def test_strips_whitespace_from_string_fields(self, build_document_chunk_payload: PayloadFactory) -> None:
        chunk = DocumentChunk.model_validate(
            build_document_chunk_payload(
                package="  context8  ",
                version="  stable  ",
                text="  Trim me.  ",
            )
        )

        assert chunk.package == "context8"
        assert chunk.version == "stable"
        assert chunk.text == "Trim me."


@pytest.mark.validation
class TestDocumentChunkValidation:
    @pytest.mark.parametrize("field_name", ["id", "package", "version", "text"])
    def test_rejects_blank_required_strings(
        self,
        build_document_chunk_payload: PayloadFactory,
        field_name: str,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk.model_validate(build_document_chunk_payload(**{field_name: "   "}))

        assert exc_info.value.errors(include_url=False)[0]["loc"] == (field_name,)

    def test_rejects_extra_fields(self, build_document_chunk_payload: PayloadFactory) -> None:
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk.model_validate(build_document_chunk_payload(unexpected="value"))

        assert exc_info.value.errors(include_url=False)[0]["type"] == "extra_forbidden"

    def test_rejects_non_strict_ordinal(self, build_document_chunk_payload: PayloadFactory) -> None:
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk.model_validate(build_document_chunk_payload(ordinal="1"))

        assert exc_info.value.errors(include_url=False)[0]["loc"] == ("ordinal",)

    def test_is_top_level_immutable(self, document_chunk_payload: dict[str, Any]) -> None:
        chunk = DocumentChunk.model_validate(document_chunk_payload)

        with pytest.raises(ValidationError):
            chunk.ordinal = 2


@pytest.mark.contract
class TestEmbedModels:
    def test_embed_request_accepts_non_empty_texts(self) -> None:
        request = EmbedRequest.model_validate({"texts": ["first text", "second text"]})

        assert request.texts == ["first text", "second text"]

    def test_embed_response_accepts_matching_dimensions(
        self,
        build_embed_response_payload: PayloadFactory,
    ) -> None:
        response = EmbedResponse.model_validate(build_embed_response_payload())

        assert response.dimensions == 3
        assert response.vectors[1] == pytest.approx([0.44, 0.55, 0.66])


@pytest.mark.validation
class TestEmbedModelValidation:
    @pytest.mark.parametrize(
        ("texts", "expected_loc"),
        [
            ([], ("texts",)),
            (["valid", "   "], ("texts", 1)),
        ],
    )
    def test_embed_request_rejects_empty_input_shapes(
        self,
        texts: list[str],
        expected_loc: tuple[str, int] | tuple[str],
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EmbedRequest.model_validate({"texts": texts})

        assert exc_info.value.errors(include_url=False)[0]["loc"] == expected_loc

    def test_embed_response_rejects_dimension_mismatch(
        self,
        build_embed_response_payload: PayloadFactory,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EmbedResponse.model_validate(
                build_embed_response_payload(vectors=[[0.11, 0.22, 0.33], [0.44, 0.55]])
            )

        assert exc_info.value.errors(include_url=False)[0]["type"] == "value_error"


@pytest.mark.contract
class TestSearchAndRerankModels:
    def test_search_candidate_accepts_finite_score(self, search_candidate_payload: dict[str, Any]) -> None:
        candidate = SearchCandidate.model_validate(search_candidate_payload)

        assert candidate.score == pytest.approx(0.97)
        assert candidate.metadata["highlights"] == ["models", "validation"]

    def test_search_response_wraps_candidate_payloads(
        self,
        candidate_payloads: list[dict[str, Any]],
    ) -> None:
        response = SearchResponse.model_validate(
            {
                "package": "context8",
                "version": "latest",
                "query": "models",
                "results": candidate_payloads,
            }
        )

        assert len(response.results) == 2
        assert response.results[1].id == "chunk-002"

    def test_rerank_response_wraps_candidates(self, candidate_payloads: list[dict[str, Any]]) -> None:
        response = RerankResponse.model_validate({"results": candidate_payloads})

        assert [candidate.id for candidate in response.results] == ["chunk-001", "chunk-002"]


@pytest.mark.validation
class TestSearchAndRerankValidation:
    @pytest.mark.parametrize("score", [float("inf"), float("-inf"), float("nan")])
    def test_search_candidate_rejects_non_finite_scores(
        self,
        build_search_candidate_payload: PayloadFactory,
        score: float,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SearchCandidate.model_validate(build_search_candidate_payload(score=score))

        assert exc_info.value.errors(include_url=False)[0]["loc"] == ("score",)

    @pytest.mark.parametrize(
        ("model_cls", "payload"),
        [
            (RerankRequest, {"query": "rerank me", "candidates": []}),
            (RerankResponse, {"results": []}),
            (
                SearchResponse,
                {"package": "context8", "version": "latest", "query": "models", "results": []},
            ),
        ],
    )
    def test_models_reject_empty_candidate_lists(self, model_cls: type[Any], payload: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            model_cls.model_validate(payload)


@pytest.mark.contract
class TestIngestReport:
    def test_builds_report_with_non_negative_chunk_count(self) -> None:
        report = IngestReport.model_validate(
            {
                "package": "context8",
                "version": "latest",
                "source": "https://example.com/docs/context8",
                "chunks": 12,
                "collection": "docs-context8-latest",
            }
        )

        assert report.chunks == 12
        assert report.collection == "docs-context8-latest"


@pytest.mark.validation
def test_ingest_report_rejects_negative_chunk_count() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IngestReport.model_validate(
            {
                "package": "context8",
                "version": "latest",
                "source": "https://example.com/docs/context8",
                "chunks": -1,
                "collection": "docs-context8-latest",
            }
        )

    assert exc_info.value.errors(include_url=False)[0]["loc"] == ("chunks",)
