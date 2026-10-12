// Grounded web UI: plain JavaScript, no framework, no build step.
//
// Security rule for this whole file: API data (answers, quotes, filenames) is
// rendered with textContent / DOM nodes, never parsed as HTML. LLM output and
// uploaded filenames are untrusted; HTML-string APIs would allow XSS.
// (tests/unit/test_web.py fails the build if those APIs appear here.)

"use strict";

const $ = (id) => document.getElementById(id);
const ADMIN_KEY_STORAGE = "grounded.adminKey";

// ---------- API helpers ----------

class ApiError extends Error {
  constructor(status, body) {
    const error = body && body.error ? body.error : {};
    super(error.message || `Request failed (HTTP ${status})`);
    this.status = status;
    this.code = error.code;
    this.requestId = error.request_id;
  }
}

async function api(path, { method = "GET", json, form, admin = false } = {}) {
  const headers = {};
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form; // browser sets the multipart boundary itself
  }
  if (admin) headers["X-API-Key"] = getAdminKey();

  const response = await fetch(path, { method, headers, body });
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, data);
  return data;
}

function describeError(err) {
  if (err instanceof ApiError) {
    const ref = err.requestId ? ` (request ${err.requestId.slice(0, 8)})` : "";
    return `${err.message}${ref}`;
  }
  return "Network error: is the server running?";
}

// ---------- Admin key (per-tab only) ----------

function getAdminKey() {
  try {
    return sessionStorage.getItem(ADMIN_KEY_STORAGE) || $("admin-key").value;
  } catch {
    return $("admin-key").value;
  }
}

function initAdminKey() {
  const input = $("admin-key");
  try {
    input.value = sessionStorage.getItem(ADMIN_KEY_STORAGE) || "";
  } catch {
    /* storage unavailable: key just isn't remembered */
  }
  input.addEventListener("input", () => {
    try {
      sessionStorage.setItem(ADMIN_KEY_STORAGE, input.value);
    } catch {
      /* ignore */
    }
  });
}

function setAdminStatus(message, ok) {
  const el = $("admin-status");
  el.textContent = message;
  el.className = `status ${ok ? "ok" : "err"}`;
}

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

// ---------- Collections & documents ----------

function currentCollectionId() {
  return $("collection").value;
}

async function loadCollections(selectId) {
  const select = $("collection");
  const collections = await api("/v1/collections");
  select.replaceChildren(
    ...collections.map((c) => {
      const option = document.createElement("option");
      option.value = c.id;
      option.textContent = c.name;
      return option;
    }),
  );
  if (selectId) select.value = selectId;
  $("collection-empty").hidden = collections.length > 0;
  await loadDocuments();
}

async function loadDocuments() {
  const list = $("documents");
  const id = currentCollectionId();
  if (!id) {
    list.replaceChildren();
    $("documents-empty").hidden = true;
    return;
  }
  const documents = await api(`/v1/collections/${id}/documents`);
  list.replaceChildren(...documents.map(renderDocument));
  $("documents-empty").hidden = documents.length > 0;
}

function renderDocument(doc) {
  const li = document.createElement("li");

  const name = document.createElement("span");
  name.className = "doc-name";
  name.textContent = doc.filename;

  const meta = document.createElement("span");
  meta.className = "doc-meta";
  const pages = doc.page_count ? `${doc.page_count}p · ` : "";
  meta.textContent = `${pages}${plural(doc.chunk_count, "chunk")}`;

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "link";
  remove.textContent = "delete";
  remove.title = `Delete ${doc.filename}`;
  remove.addEventListener("click", () => deleteDocument(doc));

  const right = document.createElement("span");
  right.append(meta, remove);
  li.append(name, right);
  return li;
}

async function deleteDocument(doc) {
  try {
    await api(`/v1/documents/${doc.id}`, { method: "DELETE", admin: true });
    setAdminStatus(`Deleted ${doc.filename}`, true);
    await loadDocuments();
  } catch (err) {
    $("admin").open = true;
    setAdminStatus(describeError(err), false);
  }
}

async function onUpload(event) {
  event.preventDefault();
  const file = $("file").files[0];
  const id = currentCollectionId();
  if (!file || !id) {
    setAdminStatus(id ? "Choose a file first." : "Create a collection first.", false);
    return;
  }
  const button = event.submitter;
  button.disabled = true;
  setAdminStatus(`Uploading ${file.name}: parsing, chunking, embedding...`, true);
  try {
    const form = new FormData();
    form.append("file", file);
    const doc = await api(`/v1/collections/${id}/documents`, {
      method: "POST",
      form,
      admin: true,
    });
    setAdminStatus(`Added ${doc.filename}: ${plural(doc.chunk_count, "chunk")}`, true);
    $("upload-form").reset();
    await loadDocuments();
  } catch (err) {
    setAdminStatus(describeError(err), false);
  } finally {
    button.disabled = false;
  }
}

async function onCreateCollection(event) {
  event.preventDefault();
  const name = $("collection-name").value.trim();
  try {
    const created = await api("/v1/collections", {
      method: "POST",
      json: { name },
      admin: true,
    });
    setAdminStatus(`Created collection ${created.name}`, true);
    $("collection-form").reset();
    await loadCollections(created.id);
  } catch (err) {
    setAdminStatus(describeError(err), false);
  }
}

// ---------- Ask ----------

const STATUS_LABELS = {
  answered: "Answered",
  partial: "Partial answer",
  insufficient_context: "Not in your documents",
};

async function onAsk(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  const id = currentCollectionId();
  const error = $("ask-error");
  error.hidden = true;
  if (!id) {
    error.textContent = "Create a collection and upload a document first.";
    error.hidden = false;
    return;
  }

  const button = $("ask-button");
  button.disabled = true;
  button.textContent = "Thinking...";
  try {
    const result = await api(`/v1/collections/${id}/query`, {
      method: "POST",
      json: { question },
    });
    renderResult(result);
  } catch (err) {
    $("result").hidden = true;
    error.textContent = describeError(err);
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

function renderResult(result) {
  $("intro").hidden = true;
  $("result").hidden = false;

  const badge = $("status-badge");
  badge.className = `badge ${result.status}`;
  badge.textContent = STATUS_LABELS[result.status] || result.status;

  // Server-written notice (e.g. "some statements could not be verified"). The
  // unverified statements themselves are never sent to the browser.
  const notice = $("answer-notice");
  notice.hidden = !result.notice;
  notice.textContent = result.notice || "";

  renderAnswer(result.answer);
  $("citations").replaceChildren(...groupBySource(result.citations).map(renderSourceCard));
  $("meta").textContent = describeUsage(result);
}

// Turn "text [S1] more [S2]" into text nodes + clickable marker buttons.
function renderAnswer(answer) {
  const container = $("answer");
  const parts = answer.split(/(\[S\d+\])/);
  container.replaceChildren(
    ...parts.map((part) => {
      const match = part.match(/^\[(S\d+)\]$/);
      if (!match) return document.createTextNode(part);
      const marker = document.createElement("button");
      marker.type = "button";
      marker.className = "marker";
      marker.textContent = match[1];
      marker.title = `Show source ${match[1]}`;
      marker.addEventListener("click", () => focusCitation(match[1]));
      return marker;
    }),
  );
}

// One card per source, even when Claude quotes the same source several times.
// (Also keeps element ids unique, so [S2] always jumps to the right card.)
function groupBySource(citations) {
  const groups = new Map();
  for (const citation of citations) {
    if (!groups.has(citation.source_id)) groups.set(citation.source_id, []);
    groups.get(citation.source_id).push(citation);
  }
  return [...groups.values()];
}

function renderSourceCard(citations) {
  const first = citations[0];
  const li = document.createElement("li");
  li.id = `cite-${first.source_id}`;

  const source = document.createElement("div");
  source.className = "cite-source";
  const tag = document.createElement("strong");
  tag.textContent = first.source_id;
  source.append(tag, ` · ${first.filename}${formatPages(first)}`);
  if (first.heading) source.append(` · ${first.heading}`);
  if (citations.length > 1) source.append(` · ${citations.length} quotes`);

  const quotes = citations.map((citation) => {
    const quote = document.createElement("blockquote");
    quote.textContent = `“${citation.quote}”`;
    return quote;
  });

  li.append(source, ...quotes);
  return li;
}

function formatPages({ page_start: start, page_end: end }) {
  if (start == null) return "";
  return start === end || end == null ? ` · p. ${start}` : ` · pp. ${start}–${end}`;
}

function focusCitation(sourceId) {
  const el = $(`cite-${sourceId}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

function describeUsage(result) {
  const seconds = (result.latency_ms / 1000).toFixed(1);
  const { claims_supported: supported, claims_rejected: rejected } = result.verification;
  const dropped = rejected ? ` (${rejected} dropped)` : "";
  const verified = `${plural(supported, "verified claim")}${dropped}`;
  if (!result.usage.llm_called) {
    const score = result.retrieval.top_score;
    const best = score == null ? "no documents" : `best match ${score.toFixed(2)}`;
    return `No relevant passages (${best}), so the AI was not called · ${seconds}s`;
  }
  const { model, input_tokens: input, output_tokens: output } = result.usage;
  return `${verified} · ${model} · ${input} in / ${output} out tokens · ${seconds}s`;
}

// ---------- Boot ----------

document.addEventListener("DOMContentLoaded", async () => {
  initAdminKey();
  $("collection").addEventListener("change", () => loadDocuments().catch(showLoadError));
  $("upload-form").addEventListener("submit", onUpload);
  $("collection-form").addEventListener("submit", onCreateCollection);
  $("ask-form").addEventListener("submit", onAsk);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("ask-form").requestSubmit();
    }
  });
  await loadCollections().catch(showLoadError);
});

function showLoadError(err) {
  const error = $("ask-error");
  error.textContent = describeError(err);
  error.hidden = false;
}
