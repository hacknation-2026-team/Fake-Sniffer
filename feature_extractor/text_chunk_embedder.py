import re
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Literal, TypeAlias

from .model_cache import HUGGINGFACE, model_dir

EmbeddingVector: TypeAlias = List[float]
InputType: TypeAlias = Literal["query", "passage", "raw"]
SourceType: TypeAlias = Literal["ocr", "webpage", "transcript", "unknown"]


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_index: int
    text: str
    embedding: EmbeddingVector
    word_count: int
    source_type: SourceType


class DocumentEmbedder:
    def __init__(
        self,
        model_name: str = "sdadas/mmlw-roberta-large",
        device: str | None = None,
        max_tokens: int = 512,
        target_words_per_chunk: int = 180,
        overlap_sentences: int = 1,
        prefixes: Dict[str, str] | None = None,
        models_dir: str | None = None,
    ) -> None:
        """
        Embeddingi dla długich i krótkich źródeł (OCR, WWW, transkrypty). Każdy logiczny
        fragment dostaje osobny wektor, żeby jedna fałszywa teza nie rozmywała się w długim tekście.

        prefixes mapuje input_type na prefiks doklejany przed enkodowaniem. Domyślne
        "query: " / "passage: " to konwencja E5. Modele sdadas/mmlw-retrieval-* chcą
        "zapytanie: " dla zapytań i nic dla pasaży: {"query": "zapytanie: ", "passage": ""}.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=str(model_dir(HUGGINGFACE, models_dir))
        )
        self.model.max_seq_length = max_tokens
        self.target_words_per_chunk = target_words_per_chunk
        self.overlap_sentences = overlap_sentences
        self.prefixes: Dict[str, str] = (
            prefixes if prefixes is not None else {"query": "query: ", "passage": "passage: "}
        )

    def _format_text(self, text: str, input_type: InputType) -> str:
        cleaned = text.strip()
        prefix = "" if input_type == "raw" else self.prefixes.get(input_type, "")
        if not prefix or cleaned.startswith(prefix):
            return cleaned
        return f"{prefix}{cleaned}"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Dzieli tekst na zdania, zachowując znaki interpunkcyjne."""
        # Normalizujemy nadmiarowe białe znaki i entery
        normalized = re.sub(r"\s+", " ", text).strip()
        # Dzielimy po kropce, wykrzykniku lub znaku zapytania, po których następuje spacja
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Dzieli tekst na chunki respektując granice zdań (z zakładką overlap_sentences).
        Jeśli tekst nie ma interpunkcji (np. surowy transkrypt ASR), tnie po słowach.
        """
        sentences = self._split_into_sentences(text)
        
        # Fallback dla surowych transkryptów bez kropek (gdy 1 "zdanie" ma > target_words)
        if len(sentences) == 1 and len(sentences[0].split()) > self.target_words_per_chunk:
            words = sentences[0].split()
            size = self.target_words_per_chunk
            step = max(1, size - min(25, size // 2))
            word_chunks: List[str] = []
            for start in range(0, len(words), step):
                word_chunks.append(" ".join(words[start : start + size]))
                # okno doszło do końca, następne byłoby tylko powtórzeniem ogona
                if start + size >= len(words):
                    break
            return word_chunks

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # Jeśli dodanie kolejnego zdania przekroczy limit i mamy już coś w buforze -> zamykamy chunk
            if current_sentences and (current_word_count + sentence_words > self.target_words_per_chunk):
                chunks.append(" ".join(current_sentences))
                
                # Zostawiamy ostatnie N zdań jako zakładkę (kontekst dla następnego chunka)
                if self.overlap_sentences > 0:
                    current_sentences = current_sentences[-self.overlap_sentences :]
                    current_word_count = sum(len(s.split()) for s in current_sentences)
                else:
                    current_sentences = []
                    current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence_words

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def extract_embeddings(
        self,
        text: str,
        source_type: SourceType = "unknown",
        input_type: InputType = "passage",
        batch_size: int = 32,
    ) -> List[EmbeddedChunk]:
        """
        Główna metoda ekstrakcji: przyjmuje cały tekst (z OCR, strony lub transkryptu),
        dzieli go na sensowne fragmenty i zwraca listę EmbeddedChunk z wektorami.
        """
        if not text or not text.strip():
            raise ValueError("Tekst wejściowy nie może być pusty.")

        raw_chunks = self._chunk_text(text)
        formatted_chunks = [self._format_text(c, input_type) for c in raw_chunks]

        vectors: np.ndarray = self.model.encode(
            formatted_chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        results: List[EmbeddedChunk] = []
        for idx, (chunk_text, vec) in enumerate(zip(raw_chunks, vectors)):
            results.append(
                EmbeddedChunk(
                    chunk_index=idx,
                    text=chunk_text,
                    embedding=[float(x) for x in vec],
                    word_count=len(chunk_text.split()),
                    source_type=source_type,
                )
            )

        return results