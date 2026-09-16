from typing import List, Dict, Set

from field_aware_rag.config import settings
from field_aware_rag.src.models import DocumentChunk, TargetSchema, RetrievedEvidence
from field_aware_rag.src.vector_store import VectorStore


class FieldAwareRetriever:
    """Executes query-bucketed retrieval per field defined in a TargetSchema."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k_per_query: int = settings.TOP_K_PER_QUERY,
        top_k_per_field: int = settings.TOP_K_PER_FIELD
    ):
        self.vector_store = vector_store
        self.top_k_per_query = top_k_per_query
        self.top_k_per_field = top_k_per_field

    def retrieve_for_schema(
        self,
        schema: TargetSchema,
        namespace: str
    ) -> Dict[str, RetrievedEvidence]:
        """
        Runs bucketed retrieval independently for every field
        in the schema within the specified Pinecone namespace.
        """

        retrieved_evidence: Dict[str, RetrievedEvidence] = {}

        for field_name, query_bucket in schema.field_buckets.items():
            evidence = self._retrieve_for_field(
                field_name=field_name,
                query_bucket=query_bucket,
                namespace=namespace
            )

            retrieved_evidence[field_name] = evidence

        return retrieved_evidence

    def _retrieve_for_field(
        self,
        field_name: str,
        query_bucket: List[str],
        namespace: str
    ) -> RetrievedEvidence:
        """
        Retrieves and deduplicates chunks across all query variants
        for a single field.
        """

        seen_chunk_ids: Set[str] = set()

        field_chunks: List[DocumentChunk] = []
        field_scores: List[float] = []
        queries_executed: List[str] = []

        for query in query_bucket:

            queries_executed.append(query)

            matches = self.vector_store.search(
                query=query,
                namespace=namespace,
                top_k=self.top_k_per_query
            )

            for chunk, score in matches:

                if chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.chunk_id)

                    field_chunks.append(chunk)
                    field_scores.append(score)

        # Sort combined results by similarity score descending
        scored_pairs = sorted(
            zip(field_chunks, field_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Keep only the strongest evidence for this field
        scored_pairs = scored_pairs[:self.top_k_per_field]

        final_chunks = [pair[0] for pair in scored_pairs]
        final_scores = [pair[1] for pair in scored_pairs]

        return RetrievedEvidence(
            field_name=field_name,
            query_used=" | ".join(queries_executed),
            chunks=final_chunks,
            similarity_scores=final_scores
        )