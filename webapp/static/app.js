"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const VERDICTS = {
  KNOWN_CAMPAIGN: ["known", "Znana kampania"],
  WEAK_SIMILARITY: ["weak", "Słabe podobieństwo"],
  NEW_CAMPAIGN_CANDIDATE: ["new", "Nowa kampania (kandydat)"],
};
const DEEPFAKE_BADGES = { high: "known", mid: "weak", low: "low", unknown: "new" };
const FEATURE_LABELS = { infrastruktura: "Infrastruktura", twarz: "Twarz", obraz: "Obraz", tekst: "Tekst" };
const PALETTE = ["#c8412b", "#2b5fb0", "#2f7d4f", "#b7791f", "#7a4fa3", "#1f8a8a", "#a33f6b", "#5d6b2f"];
const state = { files: [], selected: null, compareFrom: null, campaigns: null };

function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function svg(tag, props = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(props)) {
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  el.append(...kids.flat(Infinity).filter(Boolean));
  return el;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function fill(el, ...kids) {
  el.replaceChildren(...kids.flat(Infinity).filter(Boolean));
}

const percent = (value) => `${Math.round(value * 100)}%`;
const colorOf = (family) => PALETTE[(parseInt(String(family).split("-")[1], 10) || 0) % PALETTE.length];
const badge = (verdict) => {
  const [kind, text] = VERDICTS[verdict] || ["new", verdict];
  return h("span", { class: `badge ${kind}` }, text);
};

/* ---------- Analysis ---------- */

function chips() {
  return state.files.map((file, index) =>
    h("span", { class: "chip" }, file.name,
      h("button", { type: "button", title: "Usuń", onclick: () => { state.files.splice(index, 1); renderChips(); } }, "×")));
}

function renderChips() {
  fill(document.getElementById("chips"), chips());
}

function addFiles(list) {
  state.files.push(...list);
  renderChips();
}

function buildAnalysis(root) {
  const url = h("input", { type: "text", id: "url", placeholder: "Link do reklamy lub strony" });
  const picker = h("input", { type: "file", multiple: true, hidden: true, onchange: (e) => { addFiles([...e.target.files]); e.target.value = ""; } });
  const submit = h("button", { class: "primary", type: "submit" }, "Submit");
  const drop = h("div", { class: "drop" }, "Przeciągnij tu plik: obraz, film, audio, tekst, HTML lub HAR");
  const result = h("div", { id: "result" });

  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (e) => { e.preventDefault(); drop.classList.add("over"); });
  }
  for (const name of ["dragleave", "drop"]) {
    drop.addEventListener(name, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  }
  drop.addEventListener("drop", (e) => addFiles([...e.dataTransfer.files]));

  const form = h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); analyze(url, submit, result); } },
    h("div", { class: "row" }, url,
      h("button", { class: "plus", type: "button", title: "Dodaj plik", onclick: () => picker.click() }, "+"), picker),
    drop,
    h("div", { class: "chips", id: "chips" }),
    h("div", { class: "row", style: "justify-content:center;margin-top:14px" }, submit));

  fill(root, form, result);
  renderChips();
}

async function analyze(urlInput, submit, result) {
  const data = new FormData();
  data.append("url", urlInput.value);
  state.files.forEach((file) => data.append("files", file));
  submit.disabled = true;

  const started = Date.now();
  const timer = h("span", { class: "muted" }, "0 s");
  const tick = setInterval(() => { timer.textContent = `${Math.round((Date.now() - started) / 1000)} s`; }, 1000);
  fill(result, h("div", { class: "card" }, h("span", { class: "spinner" }), " Analizuję, pierwsze uruchomienie pobiera modele ", timer));

  try {
    renderResult(result, await api("/api/analyze", { method: "POST", body: data }));
  } catch (error) {
    fill(result, h("div", { class: "card error" }, error.message));
  } finally {
    clearInterval(tick);
    submit.disabled = false;
  }
}

function featureBars(features) {
  return h("div", { class: "features" }, features.map((f) => [
    h("span", {}, FEATURE_LABELS[f.feature] || f.feature),
    h("div", { class: "bar" }, h("span", { style: `width:${percent(Math.min(f.contribution / 0.35, 1))}` })),
    h("span", { class: "muted small" }, f.contribution.toFixed(2)),
  ]));
}

function mainInfo(r) {
  const v = r.verdict;
  if (!v) {
    return h("div", { class: "card" }, h("h2", {}, "Main info"),
      h("p", { class: "error" }, "Z tego materiału nie udało się wyciągnąć cech do porównania. Sprawdź błędy ekstraktorów poniżej."));
  }
  const nearest = v.neighbours[0];
  return h("div", { class: "card" },
    h("h2", {}, "Main info"),
    h("p", {}, badge(v.verdict), " ", `najwyższe podobieństwo ${v.best_score.toFixed(2)}`,
      nearest ? h("span", { class: "muted" }, ` do ${nearest.label} (${nearest.id})`) : null),
    v.top_features.length ? [h("h3", {}, "Co zadecydowało"), featureBars(v.top_features)] : null,
    v.reasons.length ? [h("h3", { style: "margin-top:14px" }, "Dlaczego"), h("ul", { class: "reasons" }, v.reasons.map((x) => h("li", {}, x)))] : null,
    v.outdated_clusters ? h("p", { class: "muted small" }, "Baza zmieniła się od ostatniej klasteryzacji, przelicz klastry w zakładce Campaigns.") : null);
}

function deepfakeCard(r) {
  if (!r.deepfake_enabled) return null;
  if (!r.deepfake.length) {
    return h("div", { class: "card" }, h("h2", {}, "Analiza deepfake"),
      h("p", { class: "muted" }, "Brak obrazu, filmu ani nagrania do oceny. Dodaj plik, żeby sprawdzić, czy jest syntetyczny."));
  }
  const rows = r.deepfake.map((row) => {
    const value = row.max;
    return h("tr", {},
      h("td", {}, row.item),
      h("td", {}, row.kind),
      h("td", { style: "min-width:180px" }, value == null
        ? h("span", { class: row.error ? "error" : "muted" }, row.error || "nie rozpoznano etykiety")
        : [h("div", { class: "bar" }, h("span", { style: `width:${percent(value)}` })),
           h("span", { class: "muted small" }, `max ${percent(value)}`, row.mean != null && row.mean !== value ? `, średnio ${percent(row.mean)}` : "")]),
      h("td", {}, h("span", { class: `badge ${DEEPFAKE_BADGES[row.level]}` }, row.verdict)));
  });
  return h("div", { class: "card" },
    h("h2", {}, "Analiza deepfake"),
    h("table", {}, h("thead", {}, h("tr", {}, ["Element", "Rodzaj", "Prawdopodobieństwo fałszerstwa", "Ocena"].map((t) => h("th", {}, t)))), h("tbody", {}, rows)),
    h("p", { class: "muted small", style: "margin-top:12px" },
      "To wskaźnik z gotowych klasyfikatorów, a nie dowód. Modele uczono na konkretnych zbiorach i na nowych generatorach oraz po rekompresji mylą się częściej, a głosy modele znają głównie z angielskiego."));
}

function extractorTable(steps) {
  const rows = steps.map((step) => {
    const raw = h("pre", { hidden: true }, JSON.stringify(step.data, null, 2));
    return [
      h("tr", { class: "click", onclick: () => { raw.hidden = !raw.hidden; } },
        h("td", {}, h("span", { class: `status ${step.status}` }), step.name),
        h("td", {}, step.source),
        h("td", {}, step.duration_sec != null ? `${step.duration_sec.toFixed(1)} s` : ""),
        h("td", { class: step.status === "error" ? "error" : "" }, step.summary)),
      h("tr", { hidden: false }, h("td", { colspan: 4, style: "padding:0;border:0" }, raw)),
    ];
  });
  return h("table", {}, h("thead", {}, h("tr", {}, ["Ekstraktor", "Źródło", "Czas", "Wynik"].map((t) => h("th", {}, t)))), h("tbody", {}, rows));
}

function addToBase(r) {
  const label = h("input", { type: "text", placeholder: "Opis incydentu (np. nazwa kampanii)" });
  const note = h("span", { class: "muted small" });
  const button = h("button", { type: "button", onclick: async () => {
    button.disabled = true;
    try {
      const added = await api(`/api/analyses/${r.id}/add`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label: label.value }) });
      note.textContent = `Dodano jako ${added.id}. Przelicz klastry w zakładce Campaigns.`;
    } catch (error) {
      note.textContent = error.message;
      button.disabled = false;
    }
  } }, "Dodaj do bazy");
  if (!r.has_features || r.added_as) button.disabled = true;
  return h("div", { class: "card" }, h("div", { class: "row" }, label, button), note);
}

function renderResult(root, r) {
  const v = r.verdict;
  const campaign = v && v.campaign
    ? h("span", {}, h("strong", {}, v.campaign), ` (${r.campaign_size} incydentów)`, v.family ? h("span", { class: "muted" }, `, rodzina ${v.family}`) : null)
    : h("span", { class: "muted" }, v ? "brak, nowa kampania" : "nie dotyczy");
  fill(root,
    h("div", { class: "card" },
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Plik / nazwa pliku / URL"), h("th", {}, "Przydział do kampanii"))),
        h("tbody", {}, h("tr", {}, h("td", {}, r.name), h("td", {}, campaign))))),
    deepfakeCard(r),
    mainInfo(r),
    h("div", { class: "card" }, h("h2", {}, "Użyte ekstraktory"), h("p", { class: "muted small" }, "Kliknij wiersz, żeby zobaczyć dane kroku."), extractorTable(r.steps)),
    addToBase(r));
}

/* ---------- Campaigns ---------- */

function layoutNodes(nodes, edges, radius) {
  const pos = new Map();
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length + 0.3;
    const r = nodes.length === 1 ? 0 : radius * 0.55;
    pos.set(n.id, { x: r * Math.cos(angle), y: r * Math.sin(angle) });
  });
  for (let step = 0; step < 220; step++) {
    const force = new Map(nodes.map((n) => [n.id, { x: 0, y: 0 }]));
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = pos.get(nodes[i].id), b = pos.get(nodes[j].id);
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = Math.max(dx * dx + dy * dy, 25);
        const push = 9000 / d2;
        const d = Math.sqrt(d2);
        force.get(nodes[i].id).x += (dx / d) * push; force.get(nodes[i].id).y += (dy / d) * push;
        force.get(nodes[j].id).x -= (dx / d) * push; force.get(nodes[j].id).y -= (dy / d) * push;
      }
    }
    for (const e of edges) {
      const a = pos.get(e.a), b = pos.get(e.b);
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(Math.hypot(dx, dy), 1);
      const pull = (d - radius * 0.62 * (1.25 - e.score * 0.5)) * 0.03;
      force.get(e.a).x += (dx / d) * pull; force.get(e.a).y += (dy / d) * pull;
      force.get(e.b).x -= (dx / d) * pull; force.get(e.b).y -= (dy / d) * pull;
    }
    for (const n of nodes) {
      const p = pos.get(n.id), f = force.get(n.id);
      p.x += f.x - p.x * 0.008; p.y += f.y - p.y * 0.008;
      const limit = radius * 0.8, dist = Math.hypot(p.x, p.y);
      if (dist > limit) { p.x *= limit / dist; p.y *= limit / dist; }
    }
  }
  return pos;
}

function bubble(campaign) {
  const radius = 70 + 18 * Math.sqrt(campaign.size);
  const pos = layoutNodes(campaign.nodes, campaign.edges, radius);
  const box = radius + 10;
  const title = campaign.id === "unassigned" ? "Bez kampanii (nowe)" : `Kampania ${campaign.id.split("-")[1]}`;

  const edges = campaign.edges.map((e) => {
    const a = pos.get(e.a), b = pos.get(e.b);
    return svg("line", { class: `edge${e.hard ? " hard" : ""}`, x1: a.x, y1: a.y, x2: b.x, y2: b.y, "stroke-width": 1 + 3 * e.score },
      svg("title", {}, `${e.a} – ${e.b}: ${e.score.toFixed(2)}${e.hard ? ", twardy ślad" : ""}`));
  });
  const nodes = campaign.nodes.map((n) => {
    const p = pos.get(n.id);
    const circle = svg("circle", {
      class: "node", "data-id": n.id, cx: p.x, cy: p.y, r: 11, fill: colorOf(n.family),
      onclick: () => selectNode(n.id),
    }, svg("title", {}, `${n.label} (${n.id})`));
    const label = svg("text", { class: "nodelabel", x: p.x, y: p.y + 23 }, n.label.length > 14 ? n.label.slice(0, 13) + "…" : n.label);
    return [circle, label];
  });

  const graph = svg("svg", { viewBox: `${-box} ${-box} ${2 * box} ${2 * box}` },
    svg("circle", { class: "ring", cx: 0, cy: 0, r: radius }), edges, nodes);
  return h("div", { class: "bubble" }, h("h3", {}, title), h("div", { class: "muted small" }, `${campaign.size} incydentów`), graph);
}

async function recluster(button) {
  button.disabled = true;
  try { await api("/api/cluster", { method: "POST" }); await loadCampaigns(); } catch (e) { alert(e.message); } finally { button.disabled = false; }
}

async function loadCampaigns() {
  const root = document.getElementById("campaigns");
  const data = await api("/api/campaigns");
  state.campaigns = data;
  state.selected = null;
  state.compareFrom = null;

  const button = h("button", { onclick: () => recluster(button) }, "Przelicz klastry");
  const toolbar = h("div", { class: "row", style: "justify-content:space-between;margin-bottom:14px" },
    h("span", { class: "muted" }, `${data.total} incydentów w ${data.campaigns.length} kampaniach`), button);
  const notice = data.outdated ? h("div", { class: "banner" }, "Część incydentów dodano po ostatniej klasteryzacji. Przelicz klastry, żeby trafiły do kampanii.") : null;

  const panel = h("aside", { class: "panel", id: "panel" });
  if (!data.campaigns.length) {
    fill(root, h("div", { class: "card" }, "Baza jest pusta. Zbuduj ją poleceniem build_base.py albo dodaj analizę w zakładce Analysis."));
    return;
  }
  fill(root, toolbar, notice, h("div", { class: "split" }, h("div", { class: "bubbles" }, data.campaigns.map(bubble)), panel));
  renderPanel();
}

function markSelected() {
  document.querySelectorAll(".node").forEach((n) => n.classList.toggle("selected", n.dataset.id === state.selected));
}

async function selectNode(id) {
  if (state.compareFrom && state.compareFrom !== id) {
    const other = state.compareFrom;
    state.compareFrom = null;
    state.selected = other;
    markSelected();
    await renderPanel(await api(`/api/compare?a=${encodeURIComponent(other)}&b=${encodeURIComponent(id)}`));
    return;
  }
  state.selected = id;
  state.compareFrom = null;
  markSelected();
  await renderPanel();
}

function comparisonBlock(c) {
  return h("div", { class: "compare" },
    h("h3", {}, `${c.a.label} vs ${c.b.label}`),
    h("p", {}, `Wynik podobieństwa ${c.score.toFixed(2)}`, c.duplicate ? " (duplikat lub niemal duplikat)" : "", c.hard.length ? ", twardy ślad infrastruktury" : ""),
    c.reasons.length ? h("ul", { class: "reasons" }, c.reasons.map((x) => h("li", {}, x))) : h("p", { class: "muted small" }, "Brak wspólnych śladów."));
}

async function renderPanel(comparison) {
  const panel = document.getElementById("panel");
  if (!panel) return;
  if (!state.selected) {
    fill(panel, h("h3", {}, "Szczegóły fake"), h("div", { class: "ph" }), h("p", { class: "muted" }, "Kliknij węzeł w grafie, żeby zobaczyć szczegóły."));
    return;
  }
  const incident = await api(`/api/incidents/${encodeURIComponent(state.selected)}`);
  const picture = incident.preview
    ? h("img", { class: "shot", src: `/api/incidents/${encodeURIComponent(incident.id)}/preview`, alt: incident.label })
    : h("div", { class: "ph" });
  const comparing = state.compareFrom === incident.id;
  const compareButton = h("button", { onclick: () => { state.compareFrom = comparing ? null : incident.id; renderPanel(); } },
    comparing ? "Anuluj porównanie" : "Porównaj z innym fake");
  fill(panel,
    h("h3", {}, "Szczegóły fake"), picture,
    h("strong", {}, incident.label), h("div", { class: "muted small" }, `${incident.id} · ${incident.campaign || "bez kampanii"}${incident.family ? ` · ${incident.family}` : ""}`),
    incident.profile.domains.length ? h("p", { class: "small" }, incident.profile.domains.join(", ")) : null,
    comparing ? h("div", { class: "banner", style: "margin-top:12px" }, "Kliknij drugi węzeł na grafie.") : null,
    h("div", { class: "actions" }, h("button", { onclick: () => showDetails(incident.id) }, "Więcej"), compareButton),
    comparison ? comparisonBlock(comparison) : null);
}

/* ---------- Database ---------- */

async function showDetails(id) {
  const incident = await api(`/api/incidents/${encodeURIComponent(id)}`);
  const p = incident.profile;
  const list = (items) => (items && items.length ? items.join(", ") : "–");
  const rows = [
    ["Identyfikator", incident.id], ["Opis", incident.label], ["Źródło", incident.source], ["Dodano", incident.added_at],
    ["Kampania", incident.campaign || "–"], ["Rodzina", incident.family || "–"],
    ["Domeny", list(p.domains)], ["Adresy IP", list(p.ips)], ["Sieci (ASN)", list(p.asns)],
    ["Serwery nazw", list(p.nameservers)], ["Rejestrator", p.registrar || "–"], ["Szablon DOM", p.dom_hash || "–"],
    ["Certyfikat TLS", p.cert_sha256 || "–"], ["Nazwy własne", list(p.entities)], ["Rozpoznane osoby", list(p.identities)],
    ["Hasze pHash", list(p.phashes)], ["Klatki (dHash)", String(p.dhashes)], ["Twarze", String(p.faces)],
    ["Embedding tekstu", p.has_text_embedding ? "tak" : "nie"], ["Embedding obrazu", p.has_visual_embedding ? "tak" : "nie"],
  ];
  const dialog = document.getElementById("details");
  fill(dialog,
    h("h2", {}, incident.label),
    incident.preview ? h("img", { class: "shot", style: "width:100%;max-height:240px;object-fit:cover;border-radius:8px;margin-bottom:12px", src: `/api/incidents/${encodeURIComponent(incident.id)}/preview`, alt: "" }) : null,
    h("dl", {}, rows.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    h("div", { class: "row", style: "justify-content:flex-end;margin-top:16px" }, h("button", { onclick: () => dialog.close() }, "Zamknij")));
  dialog.showModal();
}

async function loadDatabase() {
  const root = document.getElementById("database");
  const items = await api("/api/database");
  const button = h("button", { onclick: async () => { button.disabled = true; try { await api("/api/cluster", { method: "POST" }); await loadDatabase(); } catch (e) { alert(e.message); } finally { button.disabled = false; } } }, "Przelicz klastry");
  const rows = items.map((i) => h("tr", { class: "click", onclick: () => showDetails(i.id) },
    h("td", {}, i.id), h("td", {}, i.label), h("td", { class: "muted" }, i.campaign || "–"), h("td", { class: "muted" }, i.family || "–"),
    h("td", { class: "muted small" }, (i.added_at || "").replace("T", " ").slice(0, 16)),
    h("td", {}, h("button", { class: "danger", onclick: async (e) => {
      e.stopPropagation();
      if (!confirm(`Usunąć ${i.id} z bazy?`)) return;
      await api(`/api/incidents/${encodeURIComponent(i.id)}`, { method: "DELETE" });
      loadDatabase();
    } }, "Usuń"))));
  fill(root,
    h("div", { class: "row", style: "justify-content:space-between;margin-bottom:14px" }, h("span", { class: "muted" }, `${items.length} incydentów w bazie`), button),
    items.length
      ? h("div", { class: "card" }, h("table", {}, h("thead", {}, h("tr", {}, ["ID", "Opis", "Kampania", "Rodzina", "Dodano", ""].map((t) => h("th", {}, t)))), h("tbody", {}, rows)))
      : h("div", { class: "card" }, "Baza jest pusta."));
}

/* ---------- routing ---------- */

const LOADERS = { analysis: null, campaigns: loadCampaigns, database: loadDatabase };

async function show() {
  const tab = LOADERS.hasOwnProperty(location.hash.slice(1)) ? location.hash.slice(1) : "analysis";
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== tab; });
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
  try {
    if (LOADERS[tab]) await LOADERS[tab]();
  } catch (error) {
    fill(document.getElementById(tab), h("div", { class: "card error" }, error.message));
  }
}

buildAnalysis(document.getElementById("analysis"));
window.addEventListener("hashchange", show);
show();
