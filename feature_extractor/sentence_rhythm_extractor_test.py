import json
import pytest

from feature_extractor.sentence_rhythm_extractor import FEATURE_KEYS, SentenceRhythmExtractor


@pytest.fixture
def extractor() -> SentenceRhythmExtractor:
    return SentenceRhythmExtractor()


@pytest.mark.parametrize("n, words", [
    (12, "dwanaście"),
    (1200, "tysiąc dwieście"),
    (2023, "dwa tysiące dwadzieścia trzy"),
    (22000, "dwadzieścia dwa tysiące"),
    (5_000_000, "pięć milionów"),
    (-5, "minus pięć"),
])
def test_number_to_words_pl(n, words):
    assert SentenceRhythmExtractor.number_to_words_pl(n) == words


def test_number_out_of_range_raises():
    with pytest.raises(ValueError, match="zakresem"):
        SentenceRhythmExtractor.number_to_words_pl(10**12)


@pytest.mark.parametrize("token, syllables", [
    ("ciśnienie", 3),
    ("kardiologii", 4),
    ("szoku.", 2),
    ("w", 0),
    ("70%", 6),
    ("1200", 4),
    ("zł", 2),
    ("sms", 3),
])
def test_count_syllables(token, syllables):
    assert SentenceRhythmExtractor.count_syllables(token) == syllables


def test_count_words_ignores_dashes_and_punctuation():
    assert SentenceRhythmExtractor.count_words("Dziś – 70% taniej !") == 3


@pytest.mark.parametrize("token, expected", [
    ("kliknij", True),
    ("Zamów", True),
    ("zostawcie", True),
    ("czekaj", True),
    ("dzisiaj", False),
    ("zdjęcie", False),
    ("jesteście", False),
    ("70%", False),
])
def test_is_imperative(token, expected):
    assert SentenceRhythmExtractor.is_imperative(token) is expected


@pytest.mark.parametrize("text, expected", [
    (
        "Uwaga! Znany inwestor, np. Jan Kowalski, ujawnia sekret. Kliknij teraz",
        ["Uwaga!", "Znany inwestor, np. Jan Kowalski, ujawnia sekret.", "Kliknij teraz"],
    ),
    ("To kosztuje ok. 5 zł. Promocja trwa.", ["To kosztuje ok. 5 zł.", "Promocja trwa."]),
    ("Działa w 7 dni. ale tylko dziś", ["Działa w 7 dni. ale tylko dziś"]),
    ("Promocja\n\nKup teraz", ["Promocja", "Kup teraz"]),
    ("", []),
])
def test_split_sentences(text, expected):
    assert SentenceRhythmExtractor.split_sentences(text) == expected


@pytest.mark.parametrize("text, complexity, subordinate, coordinate", [
    ("Lekarze są w szoku.", "simple", [], []),
    ("Wystarczą 3 krople dziennie, a ciśnienie wraca do normy.", "coordinate", [], [", a"]),
    ("Koncerny chcą, żeby nikt się o tym nie dowiedział.", "subordinate", ["żeby"], []),
    ("Jeśli nie wierzysz, kliknij w link.", "subordinate", ["jeśli"], []),
    ("Uciskają nerwy, powodując ból.", "subordinate", ["powodując"], []),
    ("Wejdź i kliknij.", "coordinate", [], ["i kliknij"]),
    ("Kliknij w link, który widzisz poniżej, i zamów.", "multiple", ["który"], [", i"]),
])
def test_clause_analysis(extractor, text, complexity, subordinate, coordinate):
    sentence = extractor.analyze_sentence(text)

    assert sentence["complexity"] == complexity
    assert sentence["subordinate_markers"] == subordinate
    assert sentence["coordinate_markers"] == coordinate


@pytest.mark.parametrize("text, sentence_type, exclamatory", [
    ("Lekarze są w szoku!", "declarative", True),
    ("Czy to działa?", "interrogative", False),
    ("Kup teraz !", "imperative", True),
    ("Posłuchajcie.", "imperative", False),
])
def test_sentence_type(extractor, text, sentence_type, exclamatory):
    sentence = extractor.analyze_sentence(text)

    assert (sentence["type"], sentence["exclamatory"]) == (sentence_type, exclamatory)


def test_analyze_text_summary(extractor):
    result = extractor.analyze_text(
        "Uwaga! Lekarze są w szoku. Kliknij w link, który widzisz poniżej, i zamów."
    )

    assert (result["n_sentences"], result["n_words"], result["n_syllables"]) == (3, 13, 22)
    assert result["length_classes"] == {"short": 2, "medium": 1, "long": 0, "very_long": 0}
    assert result["complexity"] == {"simple": 2, "coordinate": 0, "subordinate": 0, "multiple": 1}
    assert result["readability"] == {
        "syllables_per_word": 1.69,
        "hard_word_ratio": 0.0,
        "fog_pl": 1.73,
        "pisarek": 2.17,
    }
    assert result["rhythm"]["npvi_length"] == 93.33
    assert result["rhythm"]["ends_with_imperative"] is True
    assert result["timing"] is None
    assert list(result["features"]) == list(FEATURE_KEYS)
    json.dumps(result, ensure_ascii=False)


def test_timing_from_sentence_times(extractor):
    result = extractor.analyze_sentences(
        ["Raz dwa trzy.", "Cztery pięć."], times=[(0.0, 1.5), (2.0, 3.0)]
    )
    first, second = result["sentences"]

    assert (first["duration_s"], first["pause_before_s"]) == (1.5, None)
    assert (second["pause_before_s"], second["syllables_per_second"]) == (0.5, 3.0)
    assert result["timing"]["words_per_minute"] == 120.0
    assert result["timing"]["npvi_duration"] == 40.0
    assert result["features"]["pause_mean_s"] == 0.5


def test_times_must_match_sentences(extractor):
    with pytest.raises(ValueError, match="znaczników czasu"):
        extractor.analyze_sentences(["Raz.", "Dwa."], times=[(0.0, 1.0)])


def test_punctuation_only_sentence_keeps_time_alignment(extractor):
    result = extractor.analyze_sentences(["…", "Kup teraz."], times=[(0.0, 0.5), (1.0, 2.0)])

    assert result["n_sentences"] == 1
    assert (result["sentences"][0]["index"], result["sentences"][0]["start"]) == (0, 1.0)


def test_empty_text(extractor):
    result = extractor.analyze_text("   ")

    assert result["n_sentences"] == 0
    assert result["readability"] is None
    assert all(value is None for value in result["features"].values())


@pytest.mark.parametrize("values, expected", [([10, 10, 10], 0.0), ([5, 15], 100.0), ([7], None)])
def test_npvi(values, expected):
    assert SentenceRhythmExtractor.npvi(values) == expected


def test_custom_length_limits():
    sentence = SentenceRhythmExtractor(length_limits=(3, 5, 8)).analyze_sentence("Raz dwa trzy cztery.")

    assert sentence["length_class"] == "medium"
