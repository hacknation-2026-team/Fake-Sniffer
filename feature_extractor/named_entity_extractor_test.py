import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from feature_extractor.named_entity_extractor import NamedEntityExtractor


def make_ent(text, label, lemma, start, end):
    return SimpleNamespace(text=text, label_=label, lemma_=lemma, start_char=start, end_char=end)


@pytest.fixture
def mock_nlp():
    """Podmienia spacy.load na atrapę - testy nie ładują modelu pl_core_news_lg."""
    with patch("feature_extractor.named_entity_extractor.spacy.load") as mock_load:
        nlp = MagicMock()
        mock_load.return_value = nlp
        yield nlp


def set_entities(nlp, ents):
    nlp.return_value = SimpleNamespace(ents=ents)


# --- INICJALIZACJA ---

def test_loads_requested_model(mock_nlp):
    with patch("feature_extractor.named_entity_extractor.spacy.load", return_value=mock_nlp) as mock_load:
        NamedEntityExtractor(model_name="pl_core_news_sm")
    mock_load.assert_called_once_with("pl_core_news_sm")


def test_no_ruler_without_extra_patterns(mock_nlp):
    NamedEntityExtractor()
    mock_nlp.add_pipe.assert_not_called()


def test_extra_patterns_registered_before_ner(mock_nlp):
    NamedEntityExtractor(extra_patterns=[("BRAND", "Baltic Pipeline")])

    args, kwargs = mock_nlp.add_pipe.call_args
    assert args == ("entity_ruler",)
    assert kwargs["before"] == "ner"
    assert kwargs["config"] == {"phrase_matcher_attr": "LOWER"}
    mock_nlp.add_pipe.return_value.add_patterns.assert_called_once_with(
        [{"label": "BRAND", "pattern": "Baltic Pipeline"}]
    )


# --- EKSTRAKCJA ---

def test_extract_entities_structure(mock_nlp):
    set_entities(mock_nlp, [make_ent("Donalda Tuska", "persName", "Donald Tusk", 10, 23)])
    result = NamedEntityExtractor().extract_entities("Reklama z Donalda Tuska")

    assert result == [
        {
            "text": "Donalda Tuska",
            "label": "persName",
            "lemma": "Donald Tusk",
            "start_char": 10,
            "end_char": 23,
        }
    ]


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_blank_text_returns_empty_without_calling_model(mock_nlp, text):
    assert NamedEntityExtractor().extract_entities(text) == []
    mock_nlp.assert_not_called()


def test_label_filter(mock_nlp):
    set_entities(
        mock_nlp,
        [
            make_ent("Tusk", "persName", "Tusk", 0, 4),
            make_ent("Warszawa", "placeName", "Warszawa", 10, 18),
        ],
    )
    result = NamedEntityExtractor().extract_entities("x", labels=["persName"])
    assert [e["label"] for e in result] == ["persName"]


def test_deduplication_by_label_and_lemma(mock_nlp):
    # Dwie odmiany tego samego nazwiska -> jeden wynik; ta sama fraza jako orgName zostaje osobno
    set_entities(
        mock_nlp,
        [
            make_ent("Tuska", "persName", "Tusk", 0, 5),
            make_ent("Tuskowi", "persName", "Tusk", 20, 27),
            make_ent("Tusk", "orgName", "Tusk", 40, 44),
        ],
    )
    extractor = NamedEntityExtractor()

    deduped = extractor.extract_entities("x")
    assert [(e["text"], e["label"]) for e in deduped] == [("Tuska", "persName"), ("Tusk", "orgName")]

    assert len(extractor.extract_entities("x", deduplicate=False)) == 3


def test_missing_lemma_falls_back_to_text(mock_nlp):
    set_entities(mock_nlp, [make_ent("Baltic Pipeline", "BRAND", "", 0, 15)])
    result = NamedEntityExtractor().extract_entities("x")
    assert result[0]["lemma"] == "Baltic Pipeline"


# --- KLUCZE I PODOBIEŃSTWO ---

def test_entity_keys_are_casefolded_lemmas(mock_nlp):
    set_entities(
        mock_nlp,
        [
            make_ent("Tuska", "persName", "Donald Tusk", 0, 5),
            make_ent("Baltic Pipeline", "BRAND", "Baltic Pipeline", 10, 25),
        ],
    )
    keys = NamedEntityExtractor().extract_entity_keys("x")
    assert keys == frozenset({"persName:donald tusk", "BRAND:baltic pipeline"})


def test_jaccard():
    a = frozenset({"persName:a", "orgName:b"})
    b = frozenset({"persName:a", "orgName:c"})
    assert NamedEntityExtractor.jaccard(a, b) == pytest.approx(1 / 3)
    assert NamedEntityExtractor.jaccard(a, a) == 1.0
    assert NamedEntityExtractor.jaccard(frozenset(), frozenset()) == 0.0


def test_count_by_label_counts_repeats(mock_nlp):
    set_entities(
        mock_nlp,
        [
            make_ent("Tusk", "persName", "Tusk", 0, 4),
            make_ent("Tuska", "persName", "Tusk", 10, 15),
            make_ent("Onet", "orgName", "Onet", 20, 24),
        ],
    )
    assert NamedEntityExtractor().count_by_label("x") == {"persName": 2, "orgName": 1}


# TEST INTEGRACYJNY (PRAWDZIWY MODEL)

@pytest.mark.integration
def test_real_model_finds_politician_and_custom_brand():
    extractor = NamedEntityExtractor(extra_patterns=[("BRAND", "Baltic Pipeline")])
    text = "Donald Tusk poleca platformę Baltic Pipeline, na której zarobisz tysiące."

    entities = extractor.extract_entities(text)
    labels = {(e["label"], e["lemma"].casefold()) for e in entities}

    assert ("BRAND", "baltic pipeline") in labels
    assert any(label == "persName" and "tusk" in lemma for label, lemma in labels)
