// Supplier Finder page. Plain JS, no build step. All data is inserted as text (never innerHTML).
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const result = $("#result");
  const FIELD_LABELS = {
    item: "Item", specs: "Specs", quantity: "Quantity", unit: "Unit",
    delivery_location: "Delivery city", deadline_days: "Deadline (days)",
  };
  const UNITS = ["sheets", "cft", "pcs", "kg", "m", "L", "sq ft", "jobs"];
  const NUMERIC = new Set(["quantity", "deadline_days"]);

  // ---------------------------------------------------------------- tiny DOM helper
  function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  // ---------------------------------------------------------------- API key (this browser only)
  const store = {
    get() { try { return localStorage.getItem("sf_api_key") || ""; } catch { return memKey; } },
    set(v) { memKey = v; try { localStorage.setItem("sf_api_key", v); } catch { /* private mode */ } },
  };
  let memKey = "";

  function askForKey() {
    const dlg = $("#key-dialog");
    $("#key-input").value = store.get();
    dlg.showModal();
    return new Promise((resolve) => {
      dlg.addEventListener("close", () => resolve(store.get()), { once: true });
    });
  }
  $("#key-btn").addEventListener("click", askForKey);
  $("#key-cancel").addEventListener("click", () => $("#key-dialog").close());
  $("#key-form").addEventListener("submit", () => store.set($("#key-input").value.trim()));

  // ---------------------------------------------------------------- API
  async function api(method, path, body, retried = false) {
    const res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json", "X-API-Key": store.get() },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401 && !retried) {
      await askForKey();
      return api(method, path, body, true);
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = data.error || {};
      const details = Array.isArray(err.details) ? err.details.map((d) => d.msg).join("; ") : "";
      throw new Error(`${err.message || res.statusText}${details ? ` (${details})` : ""}`);
    }
    return data;
  }

  // ---------------------------------------------------------------- views
  function setBusy(busy, text) {
    for (const b of document.querySelectorAll("button.primary")) b.disabled = busy;
    if (busy) {
      result.replaceChildren(h("div", { class: "loading", role: "status" },
        h("span", { class: "spinner", "aria-hidden": "true" }), h("span", {}, text)));
    }
  }

  function showError(message) {
    result.replaceChildren(h("p", { class: "error", role: "alert" }, message));
  }

  function understood(req) {
    if (!req) return null;
    const parts = [
      ["Item", req.item], ["Specs", req.specs],
      ["Qty", req.quantity != null ? `${+req.quantity} ${req.unit || ""}`.trim() : null],
      ["To", req.delivery_location], ["Within", req.deadline_days ? `${req.deadline_days} days` : null],
      ["Category", req.category],
    ].filter(([, v]) => v);
    return h("div", { class: "understood", "aria-label": "What was understood" },
      parts.map(([k, v]) => h("span", { class: "chip" }, h("b", {}, k), v)));
  }

  function renderClarify(req) {
    const pending = req.pending || {};
    const missing = pending.missing || [];
    const inputs = missing.map((f) => {
      const id = `f-${f}`;
      const control = f === "unit"
        ? h("select", { id, name: f }, h("option", { value: "" }, "choose…"), UNITS.map((u) => h("option", { value: u }, u)))
        : h("input", { id, name: f, type: NUMERIC.has(f) ? "number" : "text", min: NUMERIC.has(f) ? "1" : null,
                       step: f === "quantity" ? "any" : null, autocomplete: "off" });
      return h("div", {}, h("label", { for: id }, (pending.questions || {})[f] || FIELD_LABELS[f] || f), control);
    });
    const form = h("form", { class: "clarify", onsubmit: (e) => { e.preventDefault(); submitClarify(req.id, form); } },
      h("p", { class: "message" }, "A few details are missing"),
      understood(req.requirement),
      h("div", { class: "fields" }, inputs),
      h("p", { class: "or" }, "or answer in your own words"),
      h("label", { for: "f-text", class: "sr-only" }, "Answer in your own words"),
      h("textarea", { id: "f-text", name: "text", rows: "2", placeholder: "e.g. 600 metres, deliver to Jaipur within 4 weeks" }),
      h("div", { class: "row end" }, h("button", { class: "primary", type: "submit" }, "Continue")),
    );
    result.replaceChildren(form);
    form.querySelector("input, select, textarea")?.focus();
  }

  function pct(p) { return p == null ? "n/a" : `${Math.round(p * 100)}%`; }

  function vendorCard(v) {
    const conf = v.match.p_spec == null
      ? h("span", { class: "pill neutral", title: "Jev is not configured: ranked by text similarity and code checks only" }, "Not verified")
      : v.confidence === "shortlist"
        ? h("span", { class: "pill good", title: `Spec match ${pct(v.match.p_spec)}` }, "Strong match")
        : h("span", { class: "pill warn", title: "Check the listing before relying on it" }, "Check listing");
    const checks = v.checks || {};
    const why = [
      v.match.p_spec != null ? `Spec match ${pct(v.match.p_spec)} (Jev)` : "Ranked by text similarity (Jev off)",
      checks.moq_ok === true ? "minimum order fits" : checks.moq_ok === null ? "minimum order not comparable" : null,
      checks.lead_ok === true ? "delivers in time" : checks.lead_ok === null ? "no deadline given" : null,
    ].filter(Boolean).join(" · ");
    return h("li", { class: "vendor" },
      h("div", { class: "vendor-head" },
        h("span", { class: "rank", "aria-label": `Rank ${v.rank}` }, v.rank),
        h("div", { class: "vendor-title" },
          h("div", { class: "vendor-name" }, v.name),
          h("div", { class: "vendor-where" }, `${v.city}, ${v.state} · ${v.vendor_id}`)),
        conf),
      h("dl", { class: "facts" },
        h("div", {}, h("dt", {}, "Minimum order"), h("dd", {}, v.moq)),
        h("div", {}, h("dt", {}, "Lead time"), h("dd", {}, `${v.lead_time_days} days`)),
        h("div", {}, h("dt", {}, "Indicative price"), h("dd", {}, v.indicative_price || "—")),
        h("div", {}, h("dt", {}, "Payment terms"), h("dd", {}, v.payment_terms || "—"))),
      h("div", { class: "contact" },
        h("span", {}, v.contact_name || "Contact"),
        v.contact_email ? h("a", { href: `mailto:${v.contact_email}` }, v.contact_email) : null,
        v.phone ? h("a", { href: `tel:${v.phone}` }, v.phone) : null),
      h("details", { class: "why" }, h("summary", {}, "Why it matched"), h("p", {}, why), h("p", {}, v.specs)));
  }

  function renderReport(req) {
    const r = req.report;
    const kids = [
      h("p", { class: "message" }, r.message),
      understood(r.requirement),
    ];
    if (r.vendors.length) kids.push(h("ol", { class: "vendors" }, r.vendors.map(vendorCard)));
    if (r.near_misses.length) {
      kids.push(h("h2", { class: "section-title" }, `Near misses (${r.near_misses.length})`),
        h("ul", { class: "misses" }, r.near_misses.map((m) =>
          h("li", {}, h("span", { class: "miss-name" }, m.name), ` · ${m.city} · `, m.reason))));
    }
    const llm = r.summary.llm || {};
    kids.push(h("p", { class: "meta" },
      `${r.summary.candidates} candidates checked · ${llm.calls || 0} model calls · ` +
      `${(llm.input_tokens || 0) + (llm.output_tokens || 0)} tokens` +
      (llm.fallbacks ? ` · ${llm.fallbacks} model call(s) failed and fell back` : "")));
    if (llm.errors && llm.errors.length) {
      kids.push(h("details", { class: "why" }, h("summary", {}, "What failed"),
        llm.errors.map((e) => h("p", {}, `${e.step} (${e.source}): ${e.error}`))));
    }
    result.replaceChildren(...kids.filter(Boolean));
  }

  function render(req) {
    if (req.status === "clarifying") renderClarify(req);
    else if (req.status === "completed") renderReport(req);
    else if (req.status === "failed") showError(`This request failed: ${req.error || "unknown error"}`);
    else showError(`Unexpected status: ${req.status}`);
  }

  // ---------------------------------------------------------------- actions
  async function submitAsk(e) {
    e.preventDefault();
    const text = $("#ask-text").value.trim();
    if (text.length < 3) return;
    setBusy(true, "Reading the request and searching 500 vendors…");
    try { render(await api("POST", "/requests", { text })); loadHistory(); }
    catch (err) { showError(err.message); }
    finally { setBusy(false); }
  }

  async function submitClarify(id, form) {
    const answers = {};
    for (const el of form.querySelectorAll("input, select")) {
      const v = el.value.trim();
      if (!v) continue;
      answers[el.name] = NUMERIC.has(el.name) ? Number(v) : v;
    }
    const text = form.querySelector("#f-text").value.trim();
    if (!Object.keys(answers).length && !text) return;
    setBusy(true, "Updating the request and searching…");
    try { render(await api("POST", `/requests/${id}/clarify`, { answers, text: text || null })); loadHistory(); }
    catch (err) { showError(err.message); }
    finally { setBusy(false); }
  }

  const STATUS_PILL = { completed: "good", clarifying: "warn", failed: "bad", running: "neutral" };
  async function loadHistory() {
    const list = $("#history");
    if (!store.get()) return;
    try {
      const rows = await api("GET", "/requests?limit=10");
      list.replaceChildren(...(rows.length ? rows.map((r) => h("li", {},
        h("button", { type: "button", onclick: () => openRequest(r.id) },
          h("span", { class: "h-text", title: r.raw_text }, r.raw_text),
          h("span", { class: `pill ${STATUS_PILL[r.status] || "neutral"}` }, r.status)))) : [h("li", { class: "muted" }, "Nothing yet.")]));
    } catch { /* history is optional */ }
  }

  async function openRequest(id) {
    setBusy(true, "Loading…");
    try { render(await api("GET", `/requests/${id}`)); }
    catch (err) { showError(err.message); }
    finally { setBusy(false); }
  }

  $("#ask-form").addEventListener("submit", submitAsk);
  $("#ask-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#ask-form").requestSubmit();
  });
  for (const chip of document.querySelectorAll("[data-example]")) {
    chip.addEventListener("click", () => { $("#ask-text").value = chip.dataset.example; $("#ask-text").focus(); });
  }
  $("#refresh-history").addEventListener("click", loadHistory);
  loadHistory();
})();
