import re
import json
import io
from typing import List, Dict, Any, Tuple
from app.models.schemas import DocumentMetadata


class DocumentParser:
    """Utility class for cleaning, segmenting, and chunking financial documents."""

    # Common financial abbreviations to protect during sentence splitting
    _ABBREVIATIONS = [
        ("U.S.", "U_S_"),
        ("Inc.", "Inc_"),
        ("Corp.", "Corp_"),
        ("Ltd.", "Ltd_"),
        ("Co.", "Co_"),
        ("e.g.", "e_g_"),
        ("i.e.", "i_e_"),
        ("vs.", "vs_"),
        ("No.", "No_"),
        ("Vol.", "Vol_"),
        ("Sec.", "Sec_"),
    ]
    _SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"\'])')

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Removes extra whitespaces, normalizes line breaks and special characters."""
        if not text:
            return ""
        # Normalize non-breaking spaces and tabs
        text = text.replace("\u00a0", " ").replace("\t", " ")
        # Replace multiple newlines with double newline (preserve paragraphs)
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple horizontal spaces
        text = re.sub(r'[ ]{2,}', ' ', text)
        return text.strip()

    @classmethod
    def segment_sentences(cls, text: str) -> List[str]:
        """Splits financial text into individual sentence clauses."""
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []
        
        # Protect abbreviations
        protected = cleaned
        for abbr, token in cls._ABBREVIATIONS:
            protected = protected.replace(abbr, token)
        
        # First split by paragraph
        paragraphs = [p.strip() for p in protected.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            parts = cls._SENTENCE_SPLIT_REGEX.split(p)
            for s in parts:
                s_clean = s.strip()
                # Restore abbreviations
                for abbr, token in cls._ABBREVIATIONS:
                    s_clean = s_clean.replace(token, abbr)
                if len(s_clean) > 10:  # ignore trivial fragments
                    sentences.append(s_clean)
        
        # Fallback if regex found nothing
        if not sentences and cleaned:
            sentences = [cleaned]
        return sentences

    @classmethod
    def extract_metadata(cls, text: str) -> DocumentMetadata:
        """Computes statistical metadata about the document."""
        cleaned = cls.clean_text(text)
        sentences = cls.segment_sentences(cleaned)
        words = re.findall(r'\b\w+\b', cleaned)
        total_words = len(words)
        reading_time = round(total_words / 200.0, 2)  # Average 200 wpm reading speed

        return DocumentMetadata(
            total_characters=len(cleaned),
            total_words=total_words,
            total_sentences=len(sentences),
            estimated_reading_time_mins=max(0.1, reading_time)
        )

    @classmethod
    def chunk_text(cls, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        Splits long documents into overlapping sentence chunks to preserve context
        across transformer attention windows (512 tokens max).
        """
        sentences = cls.segment_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Retain overlap sentences
                overlap_words = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    s_w = len(s.split())
                    if overlap_words + s_w <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_words += s_w
                    else:
                        break
                current_chunk = overlap_chunk
                current_word_count = overlap_words

            current_chunk.append(sentence)
            current_word_count += sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @classmethod
    def parse_uploaded_file(cls, filename: str, content: bytes) -> str:
        """Extracts text content from various file formats (txt, json, csv, pdf)."""
        lower_name = filename.lower()
        if lower_name.endswith(('.txt', '.md', '.log')):
            return content.decode("utf-8", errors="replace")
        elif lower_name.endswith('.json'):
            try:
                data = json.loads(content.decode("utf-8", errors="replace"))
                if isinstance(data, dict):
                    # Combine strings or extract text fields
                    parts = []
                    for k, v in data.items():
                        if isinstance(v, str):
                            parts.append(f"{k}: {v}")
                        elif isinstance(v, list):
                            parts.append(f"{k}: {', '.join(str(i) for i in v)}")
                    return "\n".join(parts)
                elif isinstance(data, list):
                    return "\n".join(str(item) for item in data)
                return str(data)
            except Exception as e:
                return content.decode("utf-8", errors="replace")
        elif lower_name.endswith('.csv'):
            return content.decode("utf-8", errors="replace")
        elif lower_name.endswith('.pdf'):
            # Attempt to extract text using basic pdf stream extraction or pypdf if present
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except Exception:
                # Basic text extraction from raw PDF byte stream
                text_stream = content.decode("latin1", errors="ignore")
                matches = re.findall(r'\((.*?)\)Tj', text_stream)
                if matches:
                    return " ".join(matches)
                return "PDF uploaded. Raw text extraction requires pypdf or PyMuPDF."
        else:
            return content.decode("utf-8", errors="replace")
