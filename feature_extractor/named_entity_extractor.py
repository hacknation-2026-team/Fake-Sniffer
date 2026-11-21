import spacy
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple, TypedDict


class EntityDict(TypedDict):
    text: str
    label: str
    lemma: str
    start_char: int
    end_char: int


# (etykieta, fraza), np. ("BRAND", "Baltic Pipeline")
EntityPattern = Tuple[str, str]


class NamedEntityExtractor:
    def __init__(
        self,
        model_name: str = "pl_core_news_lg",
        extra_patterns: Sequence[EntityPattern] | None = None,
    ):
        """
        Nazwy własne przez spaCy. Model polski zwraca etykiety NKJP: persName, orgName,
        placeName, geogName, date, time.

        extra_patterns to własne frazy (np. znane fałszywe platformy). Są dopasowywane bez
        względu na wielkość liter i mają pierwszeństwo przed predykcją modelu.
        """
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        self.extra_patterns: List[EntityPattern] = list(extra_patterns or [])

        if self.extra_patterns:
            # ruler stoi przed "ner", bo model nie nadpisuje encji, które już ktoś oznaczył
            ruler = self.nlp.add_pipe(
                "entity_ruler",
                before="ner",
                config={"phrase_matcher_attr": "LOWER"},
            )
            ruler.add_patterns(
                [{"label": label, "pattern": pattern} for label, pattern in self.extra_patterns]
            )

    def extract_entities(
        self,
        text: str,
        labels: Iterable[str] | None = None,
        deduplicate: bool = True,
    ) -> List[EntityDict]:
        """
        Encje z tekstu. `labels` zawęża wynik do wybranych etykiet, a `deduplicate`
        zostawia pierwsze wystąpienie każdej pary (etykieta, lemat).
        """
        if not text or not text.strip():
            return []

        wanted = set(labels) if labels is not None else None
        doc = self.nlp(text)

        results: List[EntityDict] = []
        seen: Set[Tuple[str, str]] = set()
        for ent in doc.ents:
            if wanted is not None and ent.label_ not in wanted:
                continue

            lemma = (ent.lemma_ or ent.text).strip()
            key = (ent.label_, lemma.casefold())
            if deduplicate and key in seen:
                continue
            seen.add(key)

            results.append(
                {
                    "text": str(ent.text),
                    "label": str(ent.label_),
                    "lemma": lemma,
                    "start_char": int(ent.start_char),
                    "end_char": int(ent.end_char),
                }
            )
        return results

    def extract_entity_keys(
        self, text: str, labels: Iterable[str] | None = None
    ) -> FrozenSet[str]:
        """
        Zbiór kluczy "etykieta:lemat" do porównań Jaccarda między incydentami.
        Lematyzacja zlewa odmiany (Tuska, Tuskowi -> Tusk).
        """
        return frozenset(
            f"{e['label']}:{e['lemma'].casefold()}"
            for e in self.extract_entities(text, labels=labels, deduplicate=True)
        )

    @staticmethod
    def jaccard(keys_a: FrozenSet[str], keys_b: FrozenSet[str]) -> float:
        if not keys_a and not keys_b:
            return 0.0
        return len(keys_a & keys_b) / len(keys_a | keys_b)

    def count_by_label(self, text: str) -> Dict[str, int]:
        """Ile razy padła każda etykieta (bez deduplikacji) - prosty profil tekstu."""
        counts: Dict[str, int] = {}
        for ent in self.extract_entities(text, deduplicate=False):
            counts[ent["label"]] = counts.get(ent["label"], 0) + 1
        return counts
