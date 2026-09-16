import fitz  # PyMuPDF
import tiktoken
import hashlib
from typing import List
from field_aware_rag.config import settings
from field_aware_rag.src.models import DocumentChunk


class PDFIngestor:
    """Handles page-aware PDF parsing and token-sliding window chunking."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        encoding_name: str = settings.ENCODING_NAME
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def extract_and_chunk(self, pdf_path: str, doc_id: str) -> List[DocumentChunk]:
        """Extracts text per page from PDF and chunks it while preserving page numbers."""
        doc = fitz.open(pdf_path)
        chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            
            if not text:
                continue

            page_chunks = self._chunk_text(
                text=text,
                doc_id=doc_id,
                page_number=page_num + 1,
                start_index=chunk_counter
            )
            chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

        doc.close()
        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        page_number: int,
        start_index: int
    ) -> List[DocumentChunk]:
        """Performs token-based sliding window chunking on text from a single page."""
        tokens = self.tokenizer.encode(text)
        chunks: List[DocumentChunk] = []
        
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError("CHUNK_SIZE must be strictly greater than CHUNK_OVERLAP.")

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()

            if not chunk_text:
                continue

            # Deterministic unique ID generation per chunk
            chunk_seq = start_index + len(chunks)
            chunk_id_raw = f"{doc_id}_p{page_number}_c{chunk_seq}"
            chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]

            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk_{chunk_id}",
                    doc_id=doc_id,
                    text=chunk_text,
                    page_number=page_number,
                    token_count=len(chunk_tokens),
                    metadata={"page_index": page_number - 1}
                )
            )

        return chunks