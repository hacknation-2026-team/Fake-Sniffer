import pytest
from unittest.mock import MagicMock

from feature_extractor.entity_context_extractor import EntityContextExtractor, informative_entity, text_tone


def entities_in(text, *pairs):
    found = []
    for surface, label in pairs:
        start = text.index(surface)
        found.append({"text": surface, "lemma": surface, "label": label, "start_char": start, "end_char": start + len(surface)})
    return found


@pytest.fixture
def embedder():
    mock = MagicMock()
    mock.embed_texts.side_effect = lambda texts: [[1.0, float(i)] for i, _ in enumerate(texts)]
    return mock


def test_context_is_the_sentence_with_the_mention(embedder):
    text = "Wczoraj doszło do zamachu na Rafała Brzoskę. Inwestuj z InPost i zarób 5000 zł!"
    entities = entities_in(text, ("Rafała Brzoskę", "persName"), ("InPost", "orgName"))

    contexts = EntityContextExtractor(embedder).extract(text, entities)

    brzoska = contexts["persName:rafała brzoskę"]
    assert brzoska["sentences"] == ["Wczoraj doszło do zamachu na Rafała Brzoskę."]
    assert brzoska["tone"] == "negatywny" and brzoska["in_list"] is False
    assert contexts["orgName:inpost"]["tone"] == "pozytywny"
    assert brzoska["embedding"] is not None


def test_enumerations_are_marked_and_not_embedded(embedder):
    countries = "Bermuda, Cocos, Guam, Malta, Monaco, Laos, Mali"
    text = f"Wybierz kraj: {countries}\nPolecamy naszą ofertę Revolut."
    pairs = [(c, "placeName") for c in countries.split(", ")] + [("Revolut", "orgName")]

    contexts = EntityContextExtractor(embedder, list_threshold=6).extract(text, entities_in(text, *pairs))

    assert contexts["placeName:bermuda"]["in_list"] is True
    assert contexts["placeName:bermuda"]["embedding"] is None
    assert contexts["orgName:revolut"]["in_list"] is False
    embedded = embedder.embed_texts.call_args[0][0]
    assert embedded == ["Polecamy naszą ofertę Revolut."]


def test_uninformative_entities_are_skipped(embedder):
    text = "W 2026 roku Polska zmieni prawo."
    entities = entities_in(text, ("2026", "date"), ("Polska", "placeName"))

    assert EntityContextExtractor(embedder).extract(text, entities) == {}
    embedder.embed_texts.assert_not_called()


@pytest.mark.parametrize("text, tone", [
    ("Szok! Skandal w banku.", "negatywny"),
    ("Zarób pasywny dochód bez ryzyka.", "pozytywny"),
    ("Spotkanie w czwartek o 12.", "neutralny"),
])
def test_text_tone(text, tone):
    assert text_tone(text) == tone


@pytest.mark.parametrize("key, expected", [
    ("persName:rafał brzoska", True), ("orgName:inpost", True), ("placeName:polska", False),
    ("date:2026", False), ("orgName:3s", False), ("placeName:ue", False), ("persName:ab", False),
])
def test_informative_entity(key, expected):
    assert informative_entity(key) is expected


def test_bare_names_on_own_line_have_no_context(embedder):
    text = "Kraj\nBermuda\nGuam\nCocos (Keeling) Islands\nPolecamy Revolut wszystkim klientom."
    entities = entities_in(text, ("Bermuda", "placeName"), ("Cocos", "geogName"), ("Revolut", "orgName"))

    contexts = EntityContextExtractor(embedder).extract(text, entities)

    assert contexts["placeName:bermuda"]["in_list"] is True
    assert contexts["geogName:cocos"]["in_list"] is True
    assert contexts["orgName:revolut"]["in_list"] is False
