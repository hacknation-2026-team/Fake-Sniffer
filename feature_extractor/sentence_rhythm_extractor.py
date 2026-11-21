import math
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

LENGTH_CLASSES = ("short", "medium", "long", "very_long")
COMPLEXITIES = ("simple", "coordinate", "subordinate", "multiple")
SENTENCE_TYPES = ("declarative", "interrogative", "imperative")

FEATURE_KEYS = (
    "n_sentences", "mean_sentence_length", "sentence_length_cv", "npvi_length",
    "short_ratio", "long_ratio", "complex_ratio", "clauses_per_sentence",
    "question_ratio", "imperative_ratio", "exclamatory_ratio", "numbers_per_sentence",
    "syllables_per_word", "hard_word_ratio", "fog_pl", "pisarek",
    "words_per_minute", "syllables_per_second", "pause_mean_s", "npvi_duration",
)

VOWELS = frozenset("aąeęioóuy")
_LEADING = "\"'„“«‹([{"
_TRAILING = "\"'”’»›)]}"
_EDGE_PUNCT = _LEADING + _TRAILING + ".,;:!?…–—-*"

ABBREVIATIONS = frozenset({
    "np", "tzw", "tj", "tzn", "m.in", "dr", "prof", "inż", "mgr", "hab", "doc", "ul", "al", "wg", "ww",
    "jw", "pt", "p", "św", "ks", "gen", "płk", "mjr", "kpt", "por", "sp", "red", "tys", "mln", "mld",
    "zob", "ds",
})
ABBREVIATIONS_BEFORE_NUMBER = frozenset({
    "ok", "nr", "godz", "tel", "str", "art", "ust", "pkt", "poz", "s", "min",
})

SPOKEN_FORMS = {
    "zł": "złotych", "gr": "groszy", "tys": "tysięcy", "mln": "milionów", "mld": "miliardów",
    "proc": "procent", "np": "na przykład", "tzw": "tak zwany", "itd": "i tak dalej",
    "itp": "i tym podobne", "m.in": "między innymi", "dr": "doktor", "prof": "profesor",
    "pln": "złotych", "usd": "dolarów", "km": "kilometrów", "kg": "kilogramów", "ml": "mililitrów",
}

SUBORDINATORS = frozenset({
    "że", "iż", "żeby", "aby", "ażeby", "by", "jeśli", "jeżeli", "jeśliby", "jeżeliby", "gdyby", "gdy",
    "ponieważ", "gdyż", "bo", "chociaż", "choć", "choćby", "zanim", "dopóki", "skoro", "odkąd",
    "jakby", "jakoby",
    "który", "która", "które", "którego", "której", "któremu", "którym", "którą", "których", "którymi",
}) | frozenset(
    base + ending
    for base in ("aby", "żeby", "ażeby", "gdyby", "jakby", "choćby", "jeśliby", "jeżeliby")
    for ending in ("m", "ś", "śmy", "ście")
)
WEAK_SUBORDINATORS = frozenset({
    "kiedy", "gdzie", "dokąd", "skąd", "jak", "niż", "aż", "czy", "co", "czego", "czemu", "czym",
    "kto", "kogo", "komu", "kim", "ile", "dlaczego",
    "jaki", "jaka", "jakie", "jakiego", "jakiej", "jakiemu", "jakim", "jaką", "jakich", "jakimi",
})
FRONTABLE_WEAK = frozenset({"kiedy", "gdzie", "jak"})
COORDINATORS = frozenset({
    "a", "ale", "lecz", "jednak", "jednakże", "natomiast", "więc", "zatem", "dlatego", "toteż",
    "przeto", "albo", "lub", "bądź", "ani", "i", "oraz",
})

IMPERATIVES = frozenset({
    "kup", "zamów", "sprawdź", "zadzwoń", "wejdź", "pobierz", "odbierz", "wybierz", "zostaw", "zobacz",
    "napisz", "wpisz", "zapisz", "wpłać", "zapłać", "załóż", "nałóż", "połóż", "dowiedz", "weź", "zrób",
    "daj", "dołącz", "przestań", "zostań", "wstań", "otwórz", "spójrz", "patrz", "idź", "chodź", "wróć",
    "wyślij", "pomóż", "uwierz", "zmień", "zamień", "zgłoś", "przegap", "pospiesz", "pośpiesz", "spiesz",
    "śpiesz", "zarób", "potwierdź", "polub", "pomyśl", "myśl", "pij", "jedz", "stój",
})
_NOT_IMPERATIVE = frozenset({
    "dzisiaj", "tutaj", "wczoraj", "nadzwyczaj", "zwyczaj", "obyczaj", "rodzaj", "urodzaj", "tramwaj",
    "skraj", "lokaj", "szyj",
})
_PLURAL_IMPERATIVE_BEFORE_CIE = frozenset("jńźćbwżlzp")
_PARTICIPLE_EXCEPTIONS = ("miesiąc", "tysiąc", "zając")

_UNITS = ["zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć"]
_TEENS = [
    "dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście", "piętnaście",
    "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście",
]
_TENS = [
    "", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt", "sześćdziesiąt",
    "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt",
]
_HUNDREDS = [
    "", "sto", "dwieście", "trzysta", "czterysta", "pięćset", "sześćset", "siedemset",
    "osiemset", "dziewięćset",
]
_SCALES = [
    None,
    ("tysiąc", "tysiące", "tysięcy"),
    ("milion", "miliony", "milionów"),
    ("miliard", "miliardy", "miliardów"),
]


class SentenceDict(TypedDict):
    index: int
    text: str
    n_words: int
    n_syllables: int
    n_hard_words: int
    n_numbers: int
    length_class: str
    n_clauses: int
    complexity: str
    subordinate_markers: List[str]
    coordinate_markers: List[str]
    type: str
    exclamatory: bool
    imperatives: List[str]
    start: Optional[float]
    end: Optional[float]
    duration_s: Optional[float]
    words_per_second: Optional[float]
    syllables_per_second: Optional[float]
    pause_before_s: Optional[float]


def _clean(token: str) -> str:
    return token.strip(_EDGE_PUNCT).lower()


def _is_word(clean: str) -> bool:
    return any(ch.isalnum() for ch in clean)


def _below_thousand(n: int) -> List[str]:
    words = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest < 20:
        words.append(_TEENS[rest - 10])
    else:
        tens, units = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if units:
            words.append(_UNITS[units])
    return words


def _vowel_syllables(text: str) -> int:
    total = 0
    for word in re.findall(r"[a-ząćęłńóśźż]+", text.lower()):
        for i, ch in enumerate(word):
            if ch in VOWELS and not (ch == "i" and i + 1 < len(word) and word[i + 1] in VOWELS):
                total += 1
    return total


def _is_adverbial_participle(word: str) -> bool:
    return (
        len(word) >= 5
        and word.isalpha()
        and word.endswith(("ąc", "łszy"))
        and not word.endswith(_PARTICIPLE_EXCEPTIONS)
    )


def _ends_sentence(token: str, next_token: Optional[str]) -> bool:
    stripped = token.rstrip(_TRAILING)
    if stripped.endswith(("!", "?")):
        return True
    if not stripped.endswith((".", "…")):
        return False
    if next_token is None:
        return True
    following = next_token.lstrip(_LEADING)
    if stripped.endswith(".") and not stripped.endswith(".."):
        core = stripped.lstrip(_LEADING).rstrip(".").lower()
        if core in ABBREVIATIONS:
            return False
        if core in ABBREVIATIONS_BEFORE_NUMBER and following[:1].isdigit():
            return False
    return following[:1].isupper() or following[:1].isdigit()


class SentenceRhythmExtractor:
    def __init__(
        self,
        length_limits: Tuple[int, int, int] = (7, 15, 25),
        hard_word_syllables: int = 4,
    ):
        self.length_limits = length_limits
        self.hard_word_syllables = hard_word_syllables

    @staticmethod
    def count_words(text: str) -> int:
        return sum(1 for token in (text or "").split() if _is_word(_clean(token)))

    @staticmethod
    def number_to_words_pl(n: int) -> str:
        if n < 0:
            return "minus " + SentenceRhythmExtractor.number_to_words_pl(-n)
        if n == 0:
            return "zero"

        groups: List[str] = []
        scale = 0
        while n:
            n, group = divmod(n, 1000)
            if group:
                if scale >= len(_SCALES):
                    raise ValueError("Liczba poza zakresem (maksymalnie 999 999 999 999)")
                words = _below_thousand(group)
                if scale:
                    one, few, many = _SCALES[scale]
                    if group == 1:
                        words = [one]
                    else:
                        units, last_two = group % 10, group % 100
                        words.append(few if 2 <= units <= 4 and not 12 <= last_two <= 14 else many)
                groups.append(" ".join(words))
            scale += 1
        return " ".join(reversed(groups))

    @staticmethod
    def count_syllables(token: str) -> int:
        word = _clean(token)
        if not word:
            return 0
        if any(ch.isdigit() for ch in word):
            total = 0
            for digits in re.findall(r"\d+", word):
                try:
                    total += _vowel_syllables(SentenceRhythmExtractor.number_to_words_pl(int(digits)))
                except ValueError:
                    total += 2 * len(digits)
            return total + (2 if "%" in word else 0)
        if word in SPOKEN_FORMS:
            return _vowel_syllables(SPOKEN_FORMS[word])
        syllables = _vowel_syllables(word)
        if syllables == 0 and len(word) > 1:
            return len(word)
        return syllables

    @staticmethod
    def is_imperative(token: str) -> bool:
        word = _clean(token)
        if len(word) < 3 or not word.isalpha():
            return False
        if word in IMPERATIVES:
            return True
        if word in _NOT_IMPERATIVE:
            return False
        if word.endswith("cie"):
            return len(word) >= 6 and word[-4] in _PLURAL_IMPERATIVE_BEFORE_CIE
        return (
            (word.endswith(("uj", "yj")) and len(word) >= 4)
            or (word.endswith("nij") and len(word) >= 5)
            or (word.endswith("aj") and len(word) >= 5)
        )

    @staticmethod
    def segment_sentences(tokens: Sequence[str]) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        start = 0
        for i, token in enumerate(tokens):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            if _ends_sentence(token, next_token):
                ranges.append((start, i + 1))
                start = i + 1
        if start < len(tokens):
            ranges.append((start, len(tokens)))
        return ranges

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        sentences: List[str] = []
        for paragraph in re.split(r"\n\s*\n", text or ""):
            tokens = paragraph.split()
            sentences.extend(
                " ".join(tokens[i:j]) for i, j in SentenceRhythmExtractor.segment_sentences(tokens)
            )
        return sentences

    @staticmethod
    def describe(values: Sequence[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        return {
            "mean": round(mean, 2),
            "median": round(statistics.median(values), 2),
            "std": round(std, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "cv": round(std / mean, 3) if mean else 0.0,
        }

    @staticmethod
    def npvi(values: Sequence[float]) -> Optional[float]:
        if len(values) < 2:
            return None
        terms = [abs(a - b) / ((a + b) / 2) for a, b in zip(values, values[1:]) if a + b > 0]
        return round(100 * sum(terms) / len(terms), 2) if terms else 0.0

    def analyze_text(self, text: str) -> Dict[str, Any]:
        return self.analyze_sentences(self.split_sentences(text))

    def analyze_sentence(self, text: str) -> SentenceDict:
        return self._analyze_sentence(text)[0]

    def analyze_sentences(
        self,
        sentences: Sequence[str],
        times: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        if times is not None and len(times) != len(sentences):
            raise ValueError(
                f"Liczba znaczników czasu ({len(times)}) różni się od liczby zdań ({len(sentences)})"
            )

        analyzed: List[SentenceDict] = []
        spans: List[Tuple[float, float]] = []
        alpha_words = alpha_syllables = 0
        for i, text in enumerate(sentences):
            sentence, n_alpha_words, n_alpha_syllables = self._analyze_sentence(text)
            if not sentence["n_words"]:
                continue
            sentence["index"] = len(analyzed)
            analyzed.append(sentence)
            alpha_words += n_alpha_words
            alpha_syllables += n_alpha_syllables
            if times is not None:
                spans.append(times[i])

        n_sentences = len(analyzed)
        n_words = sum(s["n_words"] for s in analyzed)
        n_syllables = sum(s["n_syllables"] for s in analyzed)
        lengths = [s["n_words"] for s in analyzed]

        length_classes = {name: 0 for name in LENGTH_CLASSES}
        complexity = {name: 0 for name in COMPLEXITIES}
        types = {name: 0 for name in SENTENCE_TYPES}
        for s in analyzed:
            length_classes[s["length_class"]] += 1
            complexity[s["complexity"]] += 1
            types[s["type"]] += 1

        readability = None
        if n_sentences:
            hard = sum(s["n_hard_words"] for s in analyzed)
            mean_length, hard_percent = n_words / n_sentences, 100 * hard / n_words
            readability = {
                "syllables_per_word": round(alpha_syllables / alpha_words, 2) if alpha_words else None,
                "hard_word_ratio": round(hard / n_words, 3),
                "fog_pl": round(0.4 * (mean_length + hard_percent), 2),
                "pisarek": round(0.5 * math.sqrt(mean_length**2 + hard_percent**2), 2),
            }

        result: Dict[str, Any] = {
            "n_sentences": n_sentences,
            "n_words": n_words,
            "n_syllables": n_syllables,
            "sentence_length": self.describe(lengths),
            "length_classes": length_classes,
            "complexity": complexity,
            "sentence_types": types,
            "exclamatory": sum(s["exclamatory"] for s in analyzed),
            "readability": readability,
            "rhythm": {
                "npvi_length": self.npvi(lengths),
                "length_pattern": [s["length_class"] for s in analyzed],
                "type_pattern": [s["type"] for s in analyzed],
                "ends_with_imperative": bool(analyzed) and analyzed[-1]["type"] == "imperative",
            },
            "timing": (
                self._timing(analyzed, spans, n_words, n_syllables)
                if times is not None and n_sentences
                else None
            ),
            "sentences": analyzed,
        }
        result["features"] = self._features(result)
        return result

    def _length_class(self, n_words: int) -> str:
        for limit, name in zip(self.length_limits, LENGTH_CLASSES):
            if n_words <= limit:
                return name
        return LENGTH_CLASSES[-1]

    def _analyze_sentence(self, text: str) -> Tuple[SentenceDict, int, int]:
        normalized = re.sub(r"\s+([,;:!?.…])", r"\1", (text or "").strip())
        all_tokens = normalized.split()
        raw = [t for t in all_tokens if _is_word(_clean(t))]
        clean = [_clean(t) for t in raw]

        tail = all_tokens[-1].rstrip(_TRAILING)[-3:] if all_tokens else ""
        is_question = "?" in tail

        syllables = [self.count_syllables(t) for t in raw]
        numeric = [any(ch.isdigit() for ch in c) for c in clean]
        alpha = [i for i in range(len(raw)) if not numeric[i]]

        subordinate, coordinate = self._clause_markers(raw, clean, is_question)
        n_clauses = 1 + len(subordinate) + len(coordinate)
        if n_clauses == 1:
            complexity = "simple"
        elif n_clauses == 2:
            complexity = "subordinate" if subordinate else "coordinate"
        else:
            complexity = "multiple"

        imperatives = [c for c in clean if self.is_imperative(c)]
        sentence: SentenceDict = {
            "index": 0,
            "text": normalized,
            "n_words": len(raw),
            "n_syllables": sum(syllables),
            "n_hard_words": sum(1 for i in alpha if syllables[i] >= self.hard_word_syllables),
            "n_numbers": sum(numeric),
            "length_class": self._length_class(len(raw)),
            "n_clauses": n_clauses,
            "complexity": complexity,
            "subordinate_markers": subordinate,
            "coordinate_markers": coordinate,
            "type": "interrogative" if is_question else "imperative" if imperatives else "declarative",
            "exclamatory": "!" in tail,
            "imperatives": imperatives,
            "start": None,
            "end": None,
            "duration_s": None,
            "words_per_second": None,
            "syllables_per_second": None,
            "pause_before_s": None,
        }
        return sentence, len(alpha), sum(syllables[i] for i in alpha)

    def _clause_markers(
        self, raw: List[str], clean: List[str], is_question: bool
    ) -> Tuple[List[str], List[str]]:
        n = len(raw)
        sep_after = [t.rstrip(_TRAILING).endswith((",", ";")) for t in raw]
        semicolon_after = [t.rstrip(_TRAILING).endswith(";") for t in raw]
        subordinate: List[str] = []
        coordinate: List[str] = []
        pending_main = False

        for i, word in enumerate(clean):
            nxt = clean[i + 1] if i + 1 < n else ""
            verb = clean[i + 2] if nxt == "nie" and i + 2 < n else nxt
            if i == 0:
                fronted = (
                    word in SUBORDINATORS
                    or (word in FRONTABLE_WEAK and not is_question)
                    or _is_adverbial_participle(word)
                )
                if fronted and any(sep_after[: n - 1]):
                    subordinate.append(word)
                    pending_main = True
                continue

            after_sep = sep_after[i - 1]
            if (
                word in SUBORDINATORS
                or (after_sep and word in WEAK_SUBORDINATORS)
                or _is_adverbial_participle(word)
            ):
                subordinate.append(word)
            elif word in COORDINATORS and (after_sep or self.is_imperative(verb)):
                if not (word == "dlatego" and nxt in SUBORDINATORS):
                    span = 3 if nxt == "nie" else 2
                    coordinate.append(f", {word}" if after_sep else " ".join(clean[i : i + span]))
            elif semicolon_after[i - 1]:
                coordinate.append(";")
            elif (
                after_sep
                and (self.is_imperative(word) or (word == "nie" and self.is_imperative(nxt)))
                and not pending_main
            ):
                coordinate.append(", " + (f"nie {nxt}" if word == "nie" else word))
            if after_sep:
                pending_main = False
        return subordinate, coordinate

    def _timing(
        self,
        sentences: List[SentenceDict],
        spans: Sequence[Tuple[float, float]],
        n_words: int,
        n_syllables: int,
    ) -> Dict[str, Any]:
        durations: List[float] = []
        pauses: List[float] = []
        previous_end: Optional[float] = None
        for sentence, (start, end) in zip(sentences, spans):
            start, end = float(start), float(end)
            duration = max(end - start, 0.0)
            sentence["start"] = round(start, 3)
            sentence["end"] = round(end, 3)
            sentence["duration_s"] = round(duration, 3)
            sentence["words_per_second"] = round(sentence["n_words"] / max(duration, 1e-3), 2)
            sentence["syllables_per_second"] = round(sentence["n_syllables"] / max(duration, 1e-3), 2)
            if previous_end is not None:
                pause = max(0.0, start - previous_end)
                sentence["pause_before_s"] = round(pause, 3)
                pauses.append(pause)
            durations.append(duration)
            previous_end = end

        speech = sum(durations)
        span = float(spans[-1][1]) - float(spans[0][0])
        pause = None
        if pauses:
            pause = {
                "mean_s": round(statistics.fmean(pauses), 3),
                "max_s": round(max(pauses), 3),
                "total_s": round(sum(pauses), 3),
                "ratio": round(sum(pauses) / span, 3) if span > 0 else 0.0,
            }
        return {
            "speech_s": round(speech, 2),
            "span_s": round(span, 2),
            "words_per_minute": round(n_words / speech * 60, 1) if speech > 0 else None,
            "syllables_per_second": round(n_syllables / speech, 2) if speech > 0 else None,
            "sentence_duration": self.describe(durations),
            "pause": pause,
            "npvi_duration": self.npvi(durations),
        }

    @staticmethod
    def _features(result: Dict[str, Any]) -> Dict[str, Any]:
        n = result["n_sentences"]
        if not n:
            return {key: None for key in FEATURE_KEYS}

        classes, complexity = result["length_classes"], result["complexity"]
        types, readability, timing = result["sentence_types"], result["readability"], result["timing"]
        sentences = result["sentences"]
        return {
            "n_sentences": n,
            "mean_sentence_length": result["sentence_length"]["mean"],
            "sentence_length_cv": result["sentence_length"]["cv"],
            "npvi_length": result["rhythm"]["npvi_length"],
            "short_ratio": round(classes["short"] / n, 3),
            "long_ratio": round((classes["long"] + classes["very_long"]) / n, 3),
            "complex_ratio": round((n - complexity["simple"]) / n, 3),
            "clauses_per_sentence": round(sum(s["n_clauses"] for s in sentences) / n, 2),
            "question_ratio": round(types["interrogative"] / n, 3),
            "imperative_ratio": round(types["imperative"] / n, 3),
            "exclamatory_ratio": round(result["exclamatory"] / n, 3),
            "numbers_per_sentence": round(sum(s["n_numbers"] for s in sentences) / n, 2),
            "syllables_per_word": readability["syllables_per_word"],
            "hard_word_ratio": readability["hard_word_ratio"],
            "fog_pl": readability["fog_pl"],
            "pisarek": readability["pisarek"],
            "words_per_minute": timing["words_per_minute"] if timing else None,
            "syllables_per_second": timing["syllables_per_second"] if timing else None,
            "pause_mean_s": timing["pause"]["mean_s"] if timing and timing["pause"] else None,
            "npvi_duration": timing["npvi_duration"] if timing else None,
        }
