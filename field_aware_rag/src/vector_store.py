from typing import List, Tuple

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from config import settings
from src.models import DocumentChunk


class VectorStore:
    """
    Persistent vector store backed by Pinecone.

    Each analysis is isolated using a Pinecone namespace.
    Document chunks are embedded and stored with provenance metadata.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
    ):
        self.encoder = SentenceTransformer(model_name)

        self.client = Pinecone(
            api_key=settings.PINECONE_API_KEY
        )

        self.index = self.client.Index(
            settings.PINECONE_INDEX_NAME
        )

    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        namespace: str,
    ) -> None:
        """
        Embeds document chunks and stores them in Pinecone.

        Each chunk is stored inside the provided analysis namespace.
        """

        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vectors = []

        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "token_count": chunk.token_count,
                        **chunk.metadata,
                    },
                }
            )

        self.index.upsert(
            vectors=vectors,
            namespace=namespace,
        )

    def search(
        self,
        query: str,
        namespace: str,
        top_k: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Searches the specified analysis namespace and returns
        the most relevant document chunks with similarity scores.
        """

        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        response = self.index.query(
            namespace=namespace,
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
        )

        results = []

        for match in response.matches:

            metadata = match.metadata

            chunk = DocumentChunk(
                chunk_id=metadata["chunk_id"],
                doc_id=metadata["doc_id"],
                text=metadata["text"],
                page_number=int(metadata["page_number"]),
                token_count=int(metadata["token_count"]),
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {
                        "chunk_id",
                        "doc_id",
                        "text",
                        "page_number",
                        "token_count",
                    }
                },
            )

            results.append(
                (
                    chunk,
                    float(match.score),
                )
            )

        return results

    def delete_namespace(self, namespace: str) -> None:
        """
        Deletes all vectors associated with an analysis namespace.
        """

        self.index.delete(
            delete_all=True,
            namespace=namespace,
        )
