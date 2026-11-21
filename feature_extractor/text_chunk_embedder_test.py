import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from feature_extractor.text_chunk_embedder import DocumentEmbedder, EmbeddedChunk


# =====================================================================
# 1. TESTY JEDNOSTKOWE (SZYBKIE, Z MOCKIEM MODELU)
# =====================================================================

@pytest.fixture
def mock_st_model():
    """Podmienia SentenceTransformer na atrapę zwracającą deterministyczne wektory jednostkowe."""
    with patch("feature_extractor.text_chunk_embedder.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()

        def fake_encode(sentences, **kwargs):
            n = len(sentences)
            vecs = np.array([[1.0 + i, 0.5, 0.25, 0.1] for i in range(n)], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        mock_instance.encode.side_effect = fake_encode
        mock_cls.return_value = mock_instance
        yield mock_instance


def test_empty_text_raises_error(mock_st_model):
    """Sprawdza, czy pusty tekst lub same białe znaki rzucają ValueError."""
    embedder = DocumentEmbedder()

    with pytest.raises(ValueError, match="nie może być pusty"):
        embedder.extract_embeddings("")

    with pytest.raises(ValueError, match="nie może być pusty"):
        embedder.extract_embeddings("   \n\t  ")


def test_short_ocr_text_returns_single_chunk(mock_st_model):
    """Krótki tekst z OCR powinien zwrócić dokładnie 1 chunk z odpowiednim prefiksem i metadanymi."""
    embedder = DocumentEmbedder()
    ocr_text = "PILNE! Rząd wprowadza podatek od deszczu już od jutra."

    chunks = embedder.extract_embeddings(
        ocr_text,
        source_type="ocr",
        input_type="query",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    
    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.chunk_index == 0
    assert chunk.text == ocr_text
    assert chunk.source_type == "ocr"
    assert chunk.word_count == 9
    assert all(isinstance(x, float) for x in chunk.embedding)

    # Sprawdzamy, czy do modelu trafił tekst z prefiksem "query: "
    called_texts = mock_st_model.encode.call_args[0][0]
    assert called_texts == [f"query: {ocr_text}"]


def test_webpage_sentence_chunking_with_overlap(mock_st_model):
    """
    Sprawdza, czy długi artykuł ze strony WWW jest dzielony na granicach pełnych zdań
    oraz czy ostatnie zdanie z chunka N pojawia się jako pierwsze w chunku N+1 (overlap).
    """
    # Ustawiamy niski limit słów (12), aby łatwo wymusić podział na kilka chunków
    embedder = DocumentEmbedder(target_words_per_chunk=12, overlap_sentences=1)

    article_text = (
        "Pierwsze zdanie artykułu ma dokładnie siedem słów. "
        "Drugie zdanie ma sześć prostych słów. "
        "Trzecie zdanie również składa się z siedmiu słów. "
        "Czwarte zdanie zamyka cały nasz przykładowy artykuł."
    )

    chunks = embedder.extract_embeddings(
        article_text,
        source_type="webpage",
        input_type="passage",
    )

    # Oczekujemy podziału:
    # Chunk 0: Zdanie 1 (7 słów) -> dodanie Zdania 2 (6 słów) dałoby 13 > 12 -> zamykamy Chunk 0
    # Chunk 1: Zdanie 1 (overlap, 7 słów) + Zdanie 2 (6 słów) -> razem 13 (bo bufor miał 1 zdanie i doszło 2.) -> przy Zdaniu 3 zamykamy Chunk 1
    assert len(chunks) >= 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source_type == "webpage" for c in chunks)

    # Weryfikacja zakładki (overlap=1): ostatnie zdanie chunka 0 musi zaczynać chunk 1
    assert chunks[1].text.startswith(chunks[0].text)
    
    # Każdy chunk musi kończyć się kropką (żadne zdanie nie zostało ucięte w połowie)
    for chunk in chunks:
        assert chunk.text.endswith(".")


def test_raw_transcript_without_punctuation_fallback(mock_st_model):
    """
    Sprawdza fallback dla surowego transkryptu wideo (np. z Whispera/ASR) bez kropek:
    powinien pociąć tekst po słowach zamiast zwracać jeden gigantyczny chunk.
    """
    embedder = DocumentEmbedder(target_words_per_chunk=50, overlap_sentences=1)

    # 130 słów bez ani jednej kropki
    raw_transcript = " ".join([f"słowo{i}" for i in range(130)])

    chunks = embedder.extract_embeddings(
        raw_transcript,
        source_type="transcript",
        input_type="passage",
    )

    # Okno 50 słów, krok 25 (zakładka 25 słów): starty 0, 25, 50, 75, 100 -> ostatnie okno sięga końca
    assert len(chunks) == 5
    assert all(c.source_type == "transcript" for c in chunks)
    assert [c.word_count for c in chunks] == [50, 50, 50, 50, 30]
    # Zakładka: ostatnie 25 słów chunka N to pierwsze 25 słów chunka N+1
    assert chunks[0].text.split()[25:] == chunks[1].text.split()[:25]


def test_custom_prefixes_for_polish_retrieval_models(mock_st_model):
    """Prefiksy można podmienić (np. 'zapytanie: ' dla sdadas/mmlw-retrieval-*)."""
    embedder = DocumentEmbedder(prefixes={"query": "zapytanie: ", "passage": ""})

    embedder.extract_embeddings("Czy 5G szkodzi?", input_type="query")
    assert mock_st_model.encode.call_args[0][0] == ["zapytanie: Czy 5G szkodzi?"]

    embedder.extract_embeddings("Treść artykułu.", input_type="passage")
    assert mock_st_model.encode.call_args[0][0] == ["Treść artykułu."]


def test_raw_input_type_adds_no_prefix(mock_st_model):
    embedder = DocumentEmbedder()
    embedder.extract_embeddings("Dowolny tekst.", input_type="raw")
    assert mock_st_model.encode.call_args[0][0] == ["Dowolny tekst."]


# =====================================================================
# 2. TESTY INTEGRACYJNE (PRAWDZIWY MODEL SIECI NEURONOWEJ)
# =====================================================================

@pytest.fixture(scope="module")
def real_embedder() -> DocumentEmbedder:
    """Ładuje prawdziwy model sdadas/mmlw-roberta-large raz na sesję testową."""
    return DocumentEmbedder(
        model_name="sdadas/mmlw-roberta-large",
        device="cpu",
        target_words_per_chunk=80,
        overlap_sentences=1,
    )


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    return float(np.dot(vec_a, vec_b))


@pytest.mark.integration
def test_real_needle_in_haystack_fake_detection(real_embedder: DocumentEmbedder):
    """
    Test scenariusza wykrywania dezinformacji ("igła w stogu siana"):
    Mamy długi, neutralny transkrypt o ogrodnictwie, w którym ukryto jeden
    fałszywy akapit o technologii 5G. Sprawdzamy, czy chunk z manipulacją
    uzyska najwyższe podobieństwo do zapytania fact-checkingowego.
    """
    neutral_intro = (
        "W dzisiejszym odcinku porozmawiamy o wiosennej pielęgnacji ogrodu przydomowego. "
        "Najważniejszym krokiem w marcu jest przycinanie drzew owocowych oraz krzewów ozdobnych. "
        "Należy pamiętać o nawożeniu gleby kompostem, aby dostarczyć roślinom niezbędnych minerałów. "
        "Regularne podlewanie wczesnym rankiem pozwala uniknąć parowania wody w słoneczne dni. "
        "Warto również sprawdzić stan narzędzi ogrodniczych i naostrzyć sekatory przed sezonem. "
    ) * 3  # ~180 słów neutralnego tła

    fake_claim_paragraph = (
        "Najnowsze tajne badania potwierdzają, że nadajniki sieci 5G w miastach emitują "
        "promieniowanie zmieniające strukturę ludzkiego DNA i powodujące natychmiastową utratę pamięci. "
    )

    neutral_outro = (
        "Na koniec przypominamy o sadzeniu roślin miododajnych, które przyciągają pszczoły i trzmiele. "
        "Lawenda, szałwia oraz wrzosy doskonale sprawdzają się na słonecznych rabatach przez całe lato. "
        "Dziękujemy za uwagę i zapraszamy na kolejny odcinek naszego poradnika działkowca w przyszłym tygodniu. "
    ) * 3  # ~150 słów neutralnego zakończenia

    full_transcript = f"{neutral_intro}{fake_claim_paragraph}{neutral_outro}"

    # 1. Ekstrakcja chunków z długiego transkryptu
    transcript_chunks = real_embedder.extract_embeddings(
        full_transcript,
        source_type="transcript",
        input_type="passage",
    )
    assert len(transcript_chunks) >= 4

    # 2. Zapytanie fact-checkingowe (np. z bazy zweryfikowanych fake newsów)
    fact_check_query = "Czy sieć 5G uszkadza ludzkie DNA i powoduje zaniki pamięci?"
    query_chunk = real_embedder.extract_embeddings(
        fact_check_query,
        source_type="ocr",
        input_type="query",
    )[0]

    # 3. Liczymy podobieństwo każdego chunka do zapytania
    scored_chunks = [
        (cosine_similarity(query_chunk.embedding, chunk.embedding), chunk)
        for chunk in transcript_chunks
    ]
    # Sortujemy od najbardziej podobnego
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)

    best_score, best_chunk = scored_chunks[0]
    worst_score, _ = scored_chunks[-1]

    # Chunk z najwyższym wynikiem MUSI zawierać ukrytą manipulację o 5G
    assert "5G" in best_chunk.text
    assert "DNA" in best_chunk.text
    assert best_score > 0.78
    assert best_score > worst_score + 0.15