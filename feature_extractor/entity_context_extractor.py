import numpy as np
import re
from typing import Any, Dict, List, Sequence, Tuple, TypedDict

IGNORED_ENTITY_LABELS = frozenset({"date", "time"})
GENERIC_ENTITIES = frozenset({
    "polska", "polski", "polak", "polacy", "polka", "poland", "europa", "europejski", "ue",
    "unia europejska", "świat", "internet", "rząd", "państwo", "kraj", "miasto", "warszawa",
})
POSITIVE_STEMS = (
    "zarob", "zarab", "zysk", "dochód", "dochod", "sukces", "gratul", "wygra", "bonus", "promoc", "rabat",
    "darmow", "bezpłat", "szans", "okazj", "bogat", "inwest", "zdrow", "ulg", "skuteczn", "gwarant", "polec",
    "cud", "rewelac", "oszczęd", "nagrod", "zniżk", "pasywn", "bezpieczn", "pewn",
)
NEGATIVE_STEMS = (
    "szok", "skandal", "tragedi", "zamach", "strzel", "aresztow", "kajdank", "zakaz", "ukryw", "oszust",
    "kłam", "chorob", "śmier", "zmarł", "atak", "ofiar", "wypadk", "pilne", "alarm", "zagroż", "niebezpiecz",
    "kryzys", "afer", "przestęp", "zatrzyma", "prokurat", "wstrząs", "dramat", "koszmar", "porwa", "zabi",
    "strat", "straci", "upadł", "bankructw",
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")


class EntityContext(TypedDict):
    sentences: List[str]
    tone: str
    in_list: bool
    embedding: List[float]


def entity_key(entity: Dict[str, Any]) -> str:
    return f"{entity['label']}:{(entity.get('lemma') or entity['text']).casefold()}"


def informative_entity(key: str) -> bool:
    label, _, lemma = key.partition(":")
    lemma = lemma.strip()
    if label in IGNORED_ENTITY_LABELS or lemma in GENERIC_ENTITIES:
        return False
    if len(re.sub(r"[\d\W_]", "", lemma)) < 3 or re.search(r"\d", lemma):
        return False
    return True


def text_tone(text: str) -> str:
    words = [w.strip(".,;:!?„”\"'()«»–-").casefold() for w in (text or "").split()]
    positive = sum(1 for w in words if w.startswith(POSITIVE_STEMS))
    negative = sum(1 for w in words if w.startswith(NEGATIVE_STEMS))
    if positive > negative:
        return "pozytywny"
    if negative > positive:
        return "negatywny"
    return "neutralny"


class EntityContextExtractor:
    def __init__(
        self,
        embedder: Any,
        max_entities: int = 200,
        sentences_per_entity: int = 3,
        list_threshold: int = 6,
        max_sentence_chars: int = 400,
        label_words: int = 4,
    ):
        self.embedder = embedder
        self.max_entities = max_entities
        self.sentences_per_entity = sentences_per_entity
        self.list_threshold = list_threshold
        self.max_sentence_chars = max_sentence_chars
        self.label_words = label_words

    def _sentences(self, text: str, entities: Sequence[Dict[str, Any]]) -> List[Tuple[int, int, str, int]]:
        starts = sorted(int(e["start_char"]) for e in entities if e.get("start_char") is not None)
        spans = []
        for match in _SENTENCE_RE.finditer(text):
            sentence = " ".join(match.group(0).split())
            if not sentence:
                continue
            count = sum(1 for s in starts if match.start() <= s < match.end())
            spans.append((match.start(), match.end(), sentence, count))
        return spans

    def extract(self, text: str, entities: Sequence[Dict[str, Any]]) -> Dict[str, EntityContext]:
        spans = self._sentences(text or "", entities)
        found: Dict[str, List[Tuple[str, int]]] = {}
        for entity in entities:
            key = entity_key(entity)
            if not informative_entity(key) or entity.get("start_char") is None:
                continue
            start = int(entity["start_char"])
            span = next((s for s in spans if s[0] <= start < s[1]), None)
            if span is None:
                continue
            bucket = found.setdefault(key, [])
            if span[2] not in (s for s, _ in bucket):
                bucket.append((span[2], span[3]))

        keys = sorted(found, key=lambda k: -len(found[k]))[: self.max_entities]
        contexts: Dict[str, Dict[str, Any]] = {}
        to_embed: List[str] = []
        for key in keys:
            sentences = found[key][: self.sentences_per_entity]
            name_words = len(key.split(":", 1)[1].split())
            bare = all(
                len(s.split()) <= name_words + 1 or (len(s.split()) <= self.label_words and not s.endswith((".", "!", "?")))
                for s, _ in sentences
            )
            in_list = bare or min(count for _, count in sentences) >= self.list_threshold
            texts = [s[: self.max_sentence_chars] for s, _ in sentences]
            contexts[key] = {"sentences": texts, "tone": text_tone(" ".join(texts)), "in_list": in_list}
            if not in_list:
                to_embed.extend(t for t in texts if t not in to_embed)

        vectors = dict(zip(to_embed, self.embedder.embed_texts(to_embed))) if to_embed else {}
        for key, context in contexts.items():
            own = [vectors[t] for t in context["sentences"] if t in vectors]
            if own:
                mean = np.mean(np.asarray(own, dtype=float), axis=0)
                norm = np.linalg.norm(mean)
                context["embedding"] = [round(float(v), 5) for v in (mean / norm if norm else mean)]
            else:
                context["embedding"] = None
        return contexts
