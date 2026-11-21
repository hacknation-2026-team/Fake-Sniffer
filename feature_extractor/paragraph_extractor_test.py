import pytest
from pathlib import Path

from feature_extractor.paragraph_extractor import ParagraphExtractor


def paragraph(n_words: int) -> str:
    return " ".join(["wyraz"] * (n_words - 1)) + " koniec."


@pytest.fixture
def extractor() -> ParagraphExtractor:
    return ParagraphExtractor()


SCAM_PAGE = """<html><head><title>Kardiolog ujawnia sposób | Dziennik Przykładowy</title></head><body>
<header class="site-header"><nav><a href="#">Kraj</a> <a href="#">Zdrowie</a></nav></header>
<article class="post">
  <h1>Kardiolog ujawnia: ten sposób obniża ciśnienie</h1>
  <p class="meta">Opublikowano 12 maja, 10:32</p>
  <p>Profesor Adam Nowak przerywa milczenie i zdradza naturalny sposób na nadciśnienie.</p>
  <blockquote><p>„Ten preparat sprawił, że wyniki wróciły do normy” – mówi profesor.</p></blockquote>
  <h2>Jak to działa?</h2>
  <ul><li>rozszerza naczynia,</li><li>obniża ciśnienie,</li><li>chroni serce.</li></ul>
  <p>Producent przygotował dla czytelników rabat 70%. <a href="#order">Zamów teraz z rabatem.</a></p>
</article>
<div class="comments">
  <p>Stosuję od tygodnia i ciśnienie mam jak młoda!</p>
  <p>Lubię to · Odpowiedz · 12 min</p>
</div>
<form class="order"><label>Imię</label><input name="name"><button>Zamów</button></form>
<div style="display: none">ciśnienie nadciśnienie tabletki kardiolog</div>
<footer><p>© 2025 Dziennik Przykładowy</p></footer>
<script>document.write("<p>Tekst z JavaScriptu.</p>");</script>
</body></html>"""
SCAM_PAGE_OTHER_TEXT = (
    SCAM_PAGE.replace("Kardiolog ujawnia", "Dietetyczka zdradza")
    .replace("Adam Nowak", "Ewa Malinowska")
    .replace("na nadciśnienie", "na nadwagę")
    .replace("rabat 70%", "rabat 50%")
)
NEWS_PAGE = f"""<html><body>
<aside class="sidebar"><p>Najczęściej czytane</p></aside>
<div class="article-content">
  <h1>Sejm przyjął budżet na przyszły rok</h1>
  <p>{paragraph(60)}</p><p>{paragraph(70)}</p>
  <h2>Co dalej z ustawą</h2>
  <p>{paragraph(55)}</p><p>{paragraph(20)}</p>
</div>
</body></html>"""


@pytest.mark.parametrize("raw, declared", [
    ('<meta charset="iso-8859-2"><p>Zażółć gęślą jaźń</p>'.encode("iso-8859-2"), None),
    ("<p>Zażółć gęślą jaźń</p>".encode("iso-8859-2"), "iso-8859-2"),
    ("<p>Zażółć gęślą jaźń</p>".encode("cp1250"), None),
    (b"\xef\xbb\xbf" + "<p>Zażółć gęślą jaźń</p>".encode("utf-8"), None),
    ("<p>Zażółć gęślą jaźń</p>".encode("utf-16"), None),
])
def test_decode_html(raw, declared):
    assert "Zażółć gęślą jaźń" in ParagraphExtractor.decode_html(raw, declared)


@pytest.mark.parametrize("text, expected", [
    ("A a.\n\nB b.\nciąg dalszy.\n\n\nC.", ["A a.", "B b. ciąg dalszy.", "C."]),
    ("Linia 1.\nLinia 2.", ["Linia 1.", "Linia 2."]),
    ("Raz.\r\n\r\nDwa.", ["Raz.", "Dwa."]),
    ("  \n  ", []),
])
def test_split_paragraphs(text, expected):
    assert ParagraphExtractor.split_paragraphs(text) == expected


def test_blocks_follow_block_elements_and_double_br(extractor):
    html = """<body>
    <div>Tekst przed akapitem.<p>Akapit w <b>środku</b> zdania.</p>Tekst po akapicie.</div>
    <div>Pierwsza część.<br><br>Druga część.<br>Ta sama linia.</div>
    <script>var x = "Nie powinno się pojawić.";</script>
    <div hidden>ukryte słowa</div><!-- komentarz HTML -->
    </body>"""

    blocks = extractor.extract_blocks(html)

    assert [b["text"] for b in blocks] == [
        "Tekst przed akapitem.",
        "Akapit w środku zdania.",
        "Tekst po akapicie.",
        "Pierwsza część.",
        "Druga część. Ta sama linia.",
    ]


def test_scam_page_main_content(extractor):
    result = extractor.analyze_html(SCAM_PAGE, url="https://example.test/a")

    assert result["container"] == {"tag": "article", "id": None, "class": "post", "text_share": 1.0}
    assert result["structure"]["pattern"] == "HsQHIIIs"
    assert (result["n_paragraphs"], result["n_headings"], result["n_quotes"], result["n_list_items"]) == (
        3, 2, 1, 3,
    )
    assert [p["level"] for p in result["paragraphs"] if p["kind"] == "heading"] == [1, 2]
    assert (result["n_comments"], result["hidden_words"], result["excluded"]["form"]) == (1, 4, 1)
    assert result["source"] == {
        "type": "html",
        "url": "https://example.test/a",
        "title": "Kardiolog ujawnia sposób | Dziennik Przykładowy",
        "h1": "Kardiolog ujawnia: ten sposób obniża ciśnienie",
    }
    assert not any("JavaScriptu" in p["text"] for p in result["paragraphs"])


def test_excluded_blocks_have_reasons(extractor):
    reasons = {b["text"]: (b["kind"], b["reason"]) for b in extractor.extract_blocks(SCAM_PAGE)}

    assert reasons["Kraj Zdrowie"] == ("boilerplate", "element szablonu strony <nav>")
    assert reasons["Opublikowano 12 maja, 10:32"][0] == "boilerplate"
    assert reasons["Lubię to · Odpowiedz · 12 min"][0] == "comment"
    assert "comments" in reasons["Lubię to · Odpowiedz · 12 min"][1]
    assert reasons["Imię"] == ("form", "formularz (np. zamówienia)")
    assert reasons["© 2025 Dziennik Przykładowy"][0] == "boilerplate"


def test_same_template_with_other_text_keeps_structure(extractor):
    scam = extractor.analyze_html(SCAM_PAGE)
    other = extractor.analyze_html(SCAM_PAGE_OTHER_TEXT)
    news = extractor.analyze_html(NEWS_PAGE)

    assert other["source"]["h1"].startswith("Dietetyczka zdradza")
    assert ParagraphExtractor.structure_similarity(scam, other) == 1.0
    assert news["structure"]["pattern"] == "HmmHms"
    assert ParagraphExtractor.structure_similarity(scam, news) < 0.6


@pytest.mark.parametrize("pattern_a, pattern_b, expected", [
    ("HsmQ", "HsmQ", 1.0),
    ("Hss", "Hms", 0.667),
    ("HsmQ", "", 0.0),
])
def test_structure_similarity_patterns(pattern_a, pattern_b, expected):
    assert ParagraphExtractor.structure_similarity(pattern_a, pattern_b) == expected


def test_paragraph_length_statistics(extractor):
    html = "<body>" + "".join(f"<p>{paragraph(n)}</p>" for n in (10, 40, 100, 200)) + "</body>"

    result = extractor.analyze_html(html)

    assert [p["length_class"] for p in result["paragraphs"]] == ["short", "medium", "long", "very_long"]
    assert result["structure"]["pattern"] == "smlx"
    assert result["paragraph_words"] == {
        "mean": 87.5, "median": 70.0, "std": 72.59, "min": 10, "max": 200, "cv": 0.83,
    }
    assert result["structure"]["npvi_words"] == 90.79
    assert (result["features"]["short_ratio"], result["features"]["long_ratio"]) == (0.25, 0.5)


def test_custom_length_limits():
    result = ParagraphExtractor(length_limits=(5, 10, 20)).analyze_html(f"<p>{paragraph(8)}</p>")

    assert result["paragraphs"][0]["length_class"] == "medium"


def test_stray_paragraph_outside_content_is_excluded(extractor):
    html = (
        f"<body><div class='content'><p>{paragraph(40)}</p><p>{paragraph(40)}</p></div>"
        "<div><p>Krótki akapit poza treścią.</p></div></body>"
    )

    result = extractor.analyze_html(html)

    assert result["container"]["class"] == "content"
    assert result["n_paragraphs"] == 2
    assert result["excluded"]["outside_container"] == 1


def test_hidden_words_counted_once(extractor):
    result = extractor.analyze_html(
        "<body><p>Widoczny akapit tekstu.</p><div hidden><p>raz dwa</p>"
        "<span style='display:none'>trzy</span></div>"
        "<p aria-hidden='true'>cztery</p><p style='visibility: hidden'>pięć sześć</p></body>"
    )

    assert result["hidden_words"] == 6
    assert result["n_paragraphs"] == 1


def test_analyze_text_detects_pseudo_headings(extractor):
    result = extractor.analyze_text(
        "Nagłówek bez kropki\n\nPierwszy akapit tekstu. Drugie zdanie.\n\nZamów teraz"
    )

    assert [p["kind"] for p in result["paragraphs"]] == ["heading", "paragraph", "paragraph"]
    assert result["structure"]["pattern"] == "Hss"
    assert result["paragraphs"][1]["n_sentences"] == 2
    assert result["source"] == {"type": "text"}


def test_analyze_file_detects_html_and_text(extractor, tmp_path: Path):
    page = tmp_path / "zapis_strony"
    page.write_bytes("<html><body><p>Akapit ze strony.</p></body></html>".encode("cp1250"))
    text = tmp_path / "tekst.txt"
    text.write_text("Pierwszy akapit.\n\nDrugi akapit.", encoding="utf-8")

    assert extractor.analyze_file(str(page))["source"]["type"] == "html"
    result = extractor.analyze_file(str(text))
    assert (result["source"]["type"], result["n_paragraphs"]) == ("text", 2)
    assert result["source"]["path"] == str(text)


def test_analyze_file_missing_raises(extractor):
    with pytest.raises(FileNotFoundError, match="brak.html"):
        extractor.analyze_file("brak.html")
