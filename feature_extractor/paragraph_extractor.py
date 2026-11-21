import difflib
import os
import re
from bs4 import BeautifulSoup
from bs4.element import CData, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from .sentence_rhythm_extractor import SentenceRhythmExtractor

PARAGRAPH_LENGTH_CLASSES = ("short", "medium", "long", "very_long")
PATTERN_LETTERS = {
    "short": "s", "medium": "m", "long": "l", "very_long": "x",
    "heading": "H", "quote": "Q", "list_item": "I",
}
FORM_MAX_SHARE = 0.5
LINK_DENSITY_MAX = 0.5
MAX_EXAMPLES = 3

TEXT_KINDS = ("paragraph", "quote")
CONTENT_KINDS = ("paragraph", "quote", "list_item")
MAIN_KINDS = ("heading", "paragraph", "quote", "list_item", "caption")
EXCLUDED_KINDS = ("comment", "boilerplate", "form", "meta", "fragment", "outside_container")

BLOCK_TAGS = frozenset({
    "html", "body", "main", "article", "section", "div", "header", "footer", "nav", "aside", "form",
    "fieldset", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd", "blockquote",
    "pre", "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "figure", "figcaption",
    "address", "center", "details", "summary", "hr", "menu", "dialog",
})
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
REMOVE_TAGS = [
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed",
    "button", "select", "input", "textarea", "option", "datalist", "video", "audio", "map", "head",
]
_STRUCTURAL_TAGS = frozenset({"html", "body", "main", "article"})
_SKIPPED_STRINGS = (Comment, Doctype, Declaration, CData, ProcessingInstruction)

COMMENT_RE = re.compile(
    r"comment|koment|disqus|replies|reply|odpowied|opini|review|recenzj|testimonial", re.I
)
UNLIKELY_RE = re.compile(
    r"-ad-|\bads?\b|advert|reklam|banner|breadcrumb|cookie|rodo|gdpr|consent|footer|stopka|header|masthead|"
    r"menu|\bnav|pagination|pager|popup|modal|newsletter|related|polecane|share|social|udostepn|sidebar|"
    r"side-bar|widget|toolbar|copyright|\btags?\b|\bmeta\b|byline|author|autor|\bdate\b|timestamp",
    re.I,
)
MAYBE_RE = re.compile(r"article|artykul|body|content|tresc|entry|main|post|story|text|tekst|column", re.I)
_META_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\bopublikowan|\baktualizacj|\bautor|\bredakcj|"
    r"czas czytania|\b(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|wrze[śs]nia|"
    r"pa[źz]dziernika|listopada|grudnia)\b",
    re.I,
)
_COMMENT_UI_RE = re.compile(
    r"lubi[ęe] to|odpowiedz|\blike\b|\breply\b|\b\d+\s*(min|godz|h|dni|d)\b\.?", re.I
)
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9._-]+)""", re.I)
_HTML_SNIFF_RE = re.compile(rb"<\s*(!doctype|html|body|head|p|div|article)\b", re.I)
_TERMINAL = (".", "!", "?", "…", ":", ";")
_CLOSERS = "\"'”’»›)]} "


class BlockDict(TypedDict):
    tag: str
    kind: str
    reason: Optional[str]
    text: str
    words: int


class ParagraphDict(TypedDict):
    index: int
    kind: str
    tag: Optional[str]
    text: str
    n_words: int
    n_sentences: int
    n_chars: int
    link_density: float
    length_class: Optional[str]
    level: Optional[int]


def _attr_str(el: Tag) -> str:
    classes = el.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return f"{' '.join(classes)} {el.get('id') or ''}".strip()


def _is_hidden(el: Tag) -> bool:
    if el.has_attr("hidden") or str(el.get("aria-hidden", "")).lower() == "true":
        return True
    style = re.sub(r"\s+", "", str(el.get("style", "")).lower())
    return "display:none" in style or "visibility:hidden" in style


def _prepare(html: str) -> Tuple[BeautifulSoup, Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = " ".join(title_tag.get_text(" ").split()) if title_tag else None

    hidden = [el for el in soup.find_all(True) if _is_hidden(el)]
    hidden_ids = {id(el) for el in hidden}
    hidden_words = 0
    for el in hidden:
        if not any(id(parent) in hidden_ids for parent in el.parents):
            hidden_words += SentenceRhythmExtractor.count_words(el.get_text(" "))
            el.decompose()
    for el in soup.find_all(REMOVE_TAGS):
        if not el.decomposed:
            el.decompose()

    h1 = soup.find("h1")
    page = {
        "title": title,
        "h1": " ".join(h1.get_text(" ").split()) if h1 else None,
        "hidden_words": hidden_words,
    }
    return soup, page


def _block_anchor(node: Any) -> Tag:
    last = None
    for parent in node.parents:
        if parent.name in BLOCK_TAGS:
            return parent
        last = parent
    return last


def _in_link(node: Any, anchor: Tag) -> bool:
    for parent in node.parents:
        if parent is anchor:
            return False
        if parent.name == "a":
            return True
    return False


def _raw_blocks(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {"anchor": None, "parts": [], "link": 0, "brs": 0}

    def flush() -> None:
        text = " ".join("".join(state["parts"]).split())
        if text:
            blocks.append({"anchor": state["anchor"], "text": text, "link_chars": state["link"]})
        state.update(anchor=None, parts=[], link=0, brs=0)

    for node in soup.descendants:
        if isinstance(node, Tag):
            if node.name == "br" and state["anchor"] is not None and _block_anchor(node) is state["anchor"]:
                state["brs"] += 1
                state["parts"].append(" ")
            elif node.name == "hr":
                flush()
            continue
        if not isinstance(node, NavigableString) or isinstance(node, _SKIPPED_STRINGS):
            continue

        text = str(node)
        anchor = _block_anchor(node)
        if anchor is not state["anchor"]:
            flush()
            if not text.strip():
                continue
            state["anchor"] = anchor
        elif text.strip() and state["brs"] >= 2:
            flush()
            state["anchor"] = anchor

        if text.strip():
            state["brs"] = 0
            state["parts"].append(text)
            if _in_link(node, anchor):
                state["link"] += len(text.strip())
        else:
            state["parts"].append(" ")
    flush()
    return blocks


def _context(anchor: Tag) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "comment": None, "boilerplate": None, "form": None, "quote": False, "caption": False,
    }
    for el in [anchor, *anchor.parents]:
        if not isinstance(el, Tag) or el.name == "[document]":
            continue
        attrs = "" if el.name in _STRUCTURAL_TAGS else _attr_str(el)
        label = f"<{el.name}{' ' + attrs if attrs else ''}>"
        if context["comment"] is None and attrs and COMMENT_RE.search(attrs):
            context["comment"] = label
        if context["boilerplate"] is None:
            if el.name in ("nav", "footer", "aside", "menu"):
                context["boilerplate"] = label
            elif el.name == "header" and not any(p.name in ("article", "main") for p in el.parents):
                context["boilerplate"] = label
            elif attrs and UNLIKELY_RE.search(attrs) and not MAYBE_RE.search(attrs):
                context["boilerplate"] = label
        if el.name == "form" and context["form"] is None:
            context["form"] = el
        if el.name == "blockquote":
            context["quote"] = True
        if el.name == "figcaption":
            context["caption"] = True
    return context


def _ends_with_terminal(text: str) -> bool:
    return text.rstrip(_CLOSERS).endswith(_TERMINAL)


def _classify(
    block: Dict[str, Any], context: Dict[str, Any], small_form: bool
) -> Tuple[str, Optional[str]]:
    text, tag = block["text"], block["anchor"].name
    words = SentenceRhythmExtractor.count_words(text)
    terminal = _ends_with_terminal(text)
    link_density = block["link_chars"] / max(len(text), 1)

    if context["comment"]:
        return "comment", f"sekcja komentarzy / opinii {context['comment']}"
    if context["boilerplate"]:
        return "boilerplate", f"element szablonu strony {context['boilerplate']}"
    if small_form:
        return "form", "formularz (np. zamówienia)"
    if tag in HEADING_TAGS:
        return "heading", None
    if link_density > LINK_DENSITY_MAX and words < 25:
        return "boilerplate", f"gęste linki ({link_density:.0%} tekstu w linkach)"
    if not terminal and words < 15 and _META_RE.search(text):
        return "meta", "metryczka (data / autor / czas czytania)"
    if context["quote"]:
        return "quote", None
    if tag in ("li", "dd", "dt"):
        return "list_item", None
    if context["caption"] or tag == "figcaption":
        return "caption", None
    if words < 4 and not terminal:
        return "fragment", "krótki fragment bez interpunkcji (przycisk, etykieta)"
    return "paragraph", None


def _classified_blocks(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    blocks = _raw_blocks(soup)
    total = sum(len(b["text"]) for b in blocks) or 1
    contexts = [_context(b["anchor"]) for b in blocks]

    form_chars: Dict[int, int] = {}
    for block, context in zip(blocks, contexts):
        if context["form"] is not None:
            form_chars[id(context["form"])] = form_chars.get(id(context["form"]), 0) + len(block["text"])

    for block, context in zip(blocks, contexts):
        form = context["form"]
        small_form = form is not None and form_chars[id(form)] / total < FORM_MAX_SHARE
        block["kind"], block["reason"] = _classify(block, context, small_form)
        block["tag"] = block["anchor"].name
    return blocks


def _depth(el: Tag) -> int:
    return sum(1 for _ in el.parents)


def _inside(anchor: Tag, container: Tag) -> bool:
    return anchor is container or any(p is container for p in anchor.parents)


class ParagraphExtractor:
    def __init__(
        self,
        length_limits: Tuple[int, int, int] = (30, 80, 150),
        container_share: float = 0.8,
        rhythm: Optional[SentenceRhythmExtractor] = None,
    ):
        self.length_limits = length_limits
        self.container_share = container_share
        self.rhythm = rhythm or SentenceRhythmExtractor()

    @staticmethod
    def decode_html(raw: bytes, declared: Optional[str] = None) -> str:
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw[3:].decode("utf-8", errors="replace")
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16", errors="replace")

        candidates = [declared] if declared else []
        match = _META_CHARSET_RE.search(raw[:8192])
        if match:
            candidates.append(match.group(1).decode("ascii", "ignore"))
        candidates += ["utf-8", "cp1250"]
        for encoding in candidates:
            try:
                return raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"\n\s*\n", text) if re.search(r"\n\s*\n", text) else text.split("\n")
        return [" ".join(p.split()) for p in parts if p.strip()]

    @staticmethod
    def structure_similarity(
        result_a: Union[Dict[str, Any], str], result_b: Union[Dict[str, Any], str]
    ) -> float:
        pattern_a = result_a if isinstance(result_a, str) else result_a["structure"]["pattern"]
        pattern_b = result_b if isinstance(result_b, str) else result_b["structure"]["pattern"]
        if not pattern_a or not pattern_b:
            return 0.0
        return round(difflib.SequenceMatcher(None, pattern_a, pattern_b, autojunk=False).ratio(), 3)

    def extract_blocks(self, html: Union[str, bytes], encoding: Optional[str] = None) -> List[BlockDict]:
        if isinstance(html, bytes):
            html = self.decode_html(html, encoding)
        soup, _ = _prepare(html)
        return [
            {
                "tag": b["tag"],
                "kind": b["kind"],
                "reason": b["reason"],
                "text": b["text"],
                "words": self.rhythm.count_words(b["text"]),
            }
            for b in _classified_blocks(soup)
        ]

    def analyze_file(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {path}")
        with open(path, "rb") as file:
            raw = file.read()

        extension = os.path.splitext(path)[1].lower()
        is_html = extension in (".html", ".htm", ".xhtml") or bool(_HTML_SNIFF_RE.search(raw[:4096]))
        result = self.analyze_html(raw) if is_html else self.analyze_text(self.decode_html(raw))
        result["source"]["path"] = str(path)
        return result

    def analyze_html(
        self,
        html: Union[str, bytes],
        url: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(html, bytes):
            html = self.decode_html(html, encoding)
        soup, page = _prepare(html)
        blocks = _classified_blocks(soup)
        container_el, share = self._select_container(blocks)

        main: List[Dict[str, Any]] = []
        excluded = {kind: 0 for kind in EXCLUDED_KINDS}
        examples: Dict[str, List[Dict[str, Any]]] = {}
        for block in blocks:
            kind, reason = block["kind"], block["reason"]
            if kind in MAIN_KINDS:
                if container_el is None or _inside(block["anchor"], container_el):
                    main.append(block)
                    continue
                kind, reason = "outside_container", "poza kontenerem treści głównej"
            excluded[kind] += 1
            if len(examples.setdefault(kind, [])) < MAX_EXAMPLES:
                examples[kind].append({"text": block["text"][:80], "reason": reason})

        n_comments = sum(
            1
            for b in blocks
            if b["kind"] == "comment"
            and not _COMMENT_UI_RE.search(b["text"])
            and (_ends_with_terminal(b["text"]) or self.rhythm.count_words(b["text"]) >= 8)
        )

        container = None
        if container_el is not None:
            classes = container_el.get("class") or []
            container = {
                "tag": container_el.name,
                "id": container_el.get("id"),
                "class": " ".join(classes) if isinstance(classes, list) else classes,
                "text_share": round(share, 3),
            }
        source = {"type": "html", "url": url, "title": page["title"], "h1": page["h1"]}
        return self._summarize(
            main, source, container, excluded, examples, page["hidden_words"], n_comments
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        paragraphs = self.split_paragraphs(text)
        main = []
        for i, paragraph in enumerate(paragraphs):
            is_heading = (
                not _ends_with_terminal(paragraph)
                and self.rhythm.count_words(paragraph) <= 10
                and i < len(paragraphs) - 1
            )
            main.append(
                {
                    "text": paragraph,
                    "kind": "heading" if is_heading else "paragraph",
                    "tag": None,
                    "link_chars": 0,
                }
            )
        excluded = {kind: 0 for kind in EXCLUDED_KINDS}
        return self._summarize(main, {"type": "text"}, None, excluded, {}, 0, 0)

    def _select_container(self, blocks: List[Dict[str, Any]]) -> Tuple[Optional[Tag], Optional[float]]:
        content = [b for b in blocks if b["kind"] in CONTENT_KINDS]
        total = sum(len(b["text"]) for b in content)
        if not total:
            return None, None

        stats: Dict[int, List[Any]] = {}
        for block in content:
            for el in [block["anchor"], *block["anchor"].parents]:
                if not isinstance(el, Tag) or el.name == "[document]":
                    continue
                entry = stats.setdefault(id(el), [el, 0, 0])
                entry[1] += len(block["text"])
                entry[2] += 1

        candidates = [
            (el, chars)
            for el, chars, count in stats.values()
            if chars >= self.container_share * total and (count >= 2 or count == len(content))
        ]
        if not candidates:
            return None, None
        el, chars = max(candidates, key=lambda c: _depth(c[0]))
        return el, chars / total

    def _length_class(self, n_words: int) -> str:
        for limit, name in zip(self.length_limits, PARAGRAPH_LENGTH_CLASSES):
            if n_words <= limit:
                return name
        return PARAGRAPH_LENGTH_CLASSES[-1]

    def _summarize(
        self,
        main: List[Dict[str, Any]],
        source: Dict[str, Any],
        container: Optional[Dict[str, Any]],
        excluded: Dict[str, int],
        examples: Dict[str, List[Dict[str, Any]]],
        hidden_words: int,
        n_comments: int,
    ) -> Dict[str, Any]:
        items: List[ParagraphDict] = []
        for index, block in enumerate(main):
            text, kind, tag = block["text"], block["kind"], block.get("tag")
            n_words = self.rhythm.count_words(text)
            items.append(
                {
                    "index": index,
                    "kind": kind,
                    "tag": tag,
                    "text": text,
                    "n_words": n_words,
                    "n_sentences": max(1, len(self.rhythm.split_sentences(text))),
                    "n_chars": len(text),
                    "link_density": round(block.get("link_chars", 0) / max(len(text), 1), 3),
                    "length_class": self._length_class(n_words) if kind in TEXT_KINDS else None,
                    "level": int(tag[1]) if kind == "heading" and tag in HEADING_TAGS else None,
                }
            )

        texts = [p for p in items if p["kind"] in TEXT_KINDS]
        words = [p["n_words"] for p in texts]
        n_texts = len(texts)
        counts = {kind: sum(1 for p in items if p["kind"] == kind) for kind in MAIN_KINDS}
        classes = {
            name: sum(1 for p in texts if p["length_class"] == name) for name in PARAGRAPH_LENGTH_CLASSES
        }
        pattern = "".join(
            PATTERN_LETTERS[p["length_class"] if p["kind"] == "paragraph" else p["kind"]]
            for p in items
            if p["kind"] != "caption"
        )
        text_chars = sum(p["n_chars"] for p in texts)
        link_chars = sum(b.get("link_chars", 0) for b in main if b["kind"] in TEXT_KINDS)
        total_words = sum(p["n_words"] for p in items)
        long_count = classes["long"] + classes["very_long"]
        word_stats = self.rhythm.describe(words)
        sentence_stats = self.rhythm.describe([p["n_sentences"] for p in texts])
        single = round(sum(1 for p in texts if p["n_sentences"] == 1) / n_texts, 3) if n_texts else None
        longest = max(texts, key=lambda p: p["n_words"])["index"] if texts else None
        per_heading = round(n_texts / counts["heading"], 2) if counts["heading"] else None
        npvi_words = self.rhythm.npvi(words)
        first_words = words[0] if words else None

        return {
            "source": source,
            "container": container,
            "n_paragraphs": n_texts,
            "n_headings": counts["heading"],
            "n_list_items": counts["list_item"],
            "n_quotes": counts["quote"],
            "n_captions": counts["caption"],
            "n_comments": n_comments,
            "total_words": total_words,
            "paragraph_words": word_stats,
            "paragraph_sentences": sentence_stats,
            "paragraph_chars": self.rhythm.describe([p["n_chars"] for p in texts]),
            "length_classes": classes,
            "single_sentence_ratio": single,
            "structure": {
                "pattern": pattern,
                "npvi_words": npvi_words,
                "first_paragraph_words": first_words,
                "last_paragraph_words": words[-1] if words else None,
                "longest_paragraph_index": longest,
                "paragraphs_per_heading": per_heading,
            },
            "excluded": excluded,
            "excluded_examples": examples,
            "hidden_words": hidden_words,
            "sentence_rhythm": self.rhythm.analyze_text("\n\n".join(p["text"] for p in texts))["features"],
            "paragraphs": items,
            "features": {
                "n_paragraphs": n_texts,
                "total_words": total_words,
                "mean_paragraph_words": word_stats["mean"] if word_stats else None,
                "median_paragraph_words": word_stats["median"] if word_stats else None,
                "paragraph_words_cv": word_stats["cv"] if word_stats else None,
                "npvi_paragraph_words": npvi_words,
                "max_paragraph_words": word_stats["max"] if word_stats else None,
                "first_paragraph_words": first_words,
                "mean_paragraph_sentences": sentence_stats["mean"] if sentence_stats else None,
                "single_sentence_ratio": single,
                "short_ratio": round(classes["short"] / n_texts, 3) if n_texts else None,
                "long_ratio": round(long_count / n_texts, 3) if n_texts else None,
                "n_headings": counts["heading"],
                "paragraphs_per_heading": per_heading,
                "n_list_items": counts["list_item"],
                "n_quotes": counts["quote"],
                "quote_ratio": round(counts["quote"] / n_texts, 3) if n_texts else None,
                "n_comments": n_comments,
                "hidden_words": hidden_words,
                "link_density": round(link_chars / text_chars, 3) if text_chars else None,
            },
        }
