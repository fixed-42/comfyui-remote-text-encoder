import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { defaultServerUrl } from "./remote_te_settings.js";
import { versionWarning } from "./remote_te_version.js";

const NODE = "RemoteEncoderStatus";
const POLL_MS = 3000;
const MIN_HEIGHT = 200;

const STYLE = `
.rte-status { font: 12px sans-serif; color: var(--fg-color, #ddd); display: flex; flex-direction: column; gap: 6px;
  overflow: auto; height: 100%; box-sizing: border-box; padding: 2px 4px; }
.rte-status .rte-line { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rte-status .rte-quiet { color: var(--descrip-text, #999); }
.rte-status .rte-error { color: #e66; white-space: pre-wrap; }
.rte-status .rte-bar { position: relative; height: 10px; border-radius: 3px; background: var(--comfy-input-bg, #222);
  border: 1px solid var(--border-color, #444); overflow: hidden; }
.rte-status .rte-bar > div { position: absolute; top: 0; bottom: 0; }
.rte-status table { border-collapse: collapse; width: 100%; }
.rte-status th { text-align: left; font-weight: 600; color: var(--descrip-text, #999); padding: 2px 4px; }
.rte-status td { padding: 2px 4px; border-top: 1px solid var(--border-color, #333); white-space: nowrap; }
.rte-status td.rte-file { white-space: normal; word-break: break-all; }
.rte-status td.rte-can span { margin-right: 6px; cursor: help; }
.rte-status .rte-can-yes { color: #4a8; }
.rte-status .rte-can-no { color: #e66; }
.rte-status .rte-can-expected { opacity: 0.6; }
.rte-status .rte-can-unknown { color: var(--descrip-text, #999); }
.rte-status select { background: var(--comfy-input-bg, #222); color: inherit; border: 1px solid var(--border-color, #444);
  font: inherit; max-width: 130px; }
.rte-status button { font: inherit; padding: 2px 10px; }
.rte-status .rte-foot { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
`;

function addStyle() {
  if (document.getElementById("rte-status-style")) return;
  const s = document.createElement("style");
  s.id = "rte-status-style";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e[k] = v;
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

const gb = (mb) => `${(mb / 1024).toFixed(1)} GB`;

const isLocal = (url) => (url ?? "").trim().toLowerCase() === "local";

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function rowsFromServer(data) {
  const rows = {};
  for (const m of data.models ?? []) {
    const kept = m.kept_as ?? [];
    const loaded = (m.loaded ?? []).map((l) => l.type);
    rows[m.file] = { keep: kept.length > 0, type: kept[0] ?? m.picked ?? loaded[0] ?? (m.types ?? [])[0] ?? "" };
  }
  return rows;
}

function plan(state) {
  const d = state.data;
  const rows = state.draft ?? rowsFromServer(d);
  let mb = 0;
  for (const m of d.models ?? []) if (rows[m.file]?.keep) mb += m.size_mb;
  return { mb, need: mb + (d.reserve_mb ?? 0), usable: d.usable_mb ?? 0, rows };
}

function canDo(m, type) {
  const checked = (type && m.can?.[type]) || {};
  return { text: checked.text ?? m.expected?.text, vision: checked.vision ?? m.expected?.vision };
}

const CAN_LABEL = { text: "Generate Text", vision: "read images (Remote Locate)" };

function unknownWhy(what, can, scan) {
  if (what === "vision" && can.text && !can.text.ok)
    return "not checked: reading images is asked through text generation, which this file can't do";
  if (!scan) return "not checked yet (checked at first use)";
  if (scan.done < scan.total) return "not checked yet (the server checks its files in the background while it is idle)";
  return "not checked: the background scan left it out (no known type, no room on the card, or it can't be " +
         "decided; the server's log says which); checked when it is used";
}

function scanLine(scan) {
  const file = scan.now?.replace(/ as [^ ]+$/, "");
  return file ? `checking what the files can do: ${scan.done + 1} of ${scan.total} (${file})`
              : `checking what the files can do: ${scan.done} of ${scan.total} done, goes on when the server is idle`;
}

function capabilityCell(m, type, scan) {
  const can = canDo(m, type);
  const cell = el("td", { class: "rte-can" });
  for (const what of ["text", "vision"]) {
    const r = can[what];
    const mark = !r ? "?" : r.ok ? "\u2713" : "\u2717";
    const cls = !r ? "rte-can-unknown" : r.ok ? "rte-can-yes" : r.checked ? "rte-can-no" : "rte-can-no rte-can-expected";
    const title = !r ? `${CAN_LABEL[what]}: ${unknownWhy(what, can, scan)}`
                : r.ok ? `${CAN_LABEL[what]}: yes (checked)` : `${CAN_LABEL[what]}: no: ${r.why}`;
    cell.append(el("span", { class: cls, title }, `${what} ${mark}`));
  }
  return cell;
}

function editing(state) {
  const a = document.activeElement;
  return a?.tagName === "SELECT" && state.root.contains(a);
}

function render(node) {
  const state = node.__rteStatus;
  const root = state.root;
  root.replaceChildren();
  const url = widget(node, "server_url")?.value ?? "";
  const local = isLocal(url); // only this ComfyUI's own server has controls
  if (!local) state.draft = null;
  if (state.error) {
    root.append(el("div", { class: "rte-error" }, state.error));
    if (!state.data) return;
  }
  if (state.notice) {
    root.append(el("div", { class: "rte-quiet" }, state.notice));
    if (!state.data) return;
  }
  const d = state.data;
  if (!d) {
    root.append(el("div", { class: "rte-quiet" }, `asking ${url} ...`));
    return;
  }
  if (state.versionWarning) root.append(el("div", { class: "rte-error" }, state.versionWarning));
  if (d.mode !== "on_demand") {
    root.append(el("div", { class: "rte-line" }, `${url}: preloaded server (router or --config), ${d.gpu ?? ""}`),
                el("div", { class: "rte-quiet" }, "Its models are set in its config file; nothing to keep here."));
    for (const m of d.models ?? []) root.append(el("div", { class: "rte-line" }, `${m.name}  (${m.file}, ${m.type})`));
    return;
  }
  const p = plan(state);
  const used = d.vram?.allocated_mb ?? 0;
  const total = d.usable_mb || 1;
  const busy = d.busy ? `busy: ${d.busy.op}${d.busy.model ? " " + d.busy.model : ""} for ${Math.round(d.busy.for_s)} s`
                      : "online";
  const card = d.gpus?.[0] ?? {};
  const shared = card.render ? `  ·  shared with ${local ? "this ComfyUI" : "that PC's ComfyUI"}` : "";
  root.append(el("div", { class: "rte-line", title: card.uuid ?? "" },
                  `${d.gpu ?? card.name ?? "GPU"}  ·  ${gb(d.total_mb ?? 0)}  ·  ${busy}${shared}`));
  const bar = el("div", { class: "rte-bar", title: "solid: in VRAM now · outline: kept models + reserve" });
  bar.append(el("div", { style: `left:0;width:${Math.min(100, (100 * used) / total)}%;background:#4a8;` }),
             el("div", { style: `left:0;width:${Math.min(100, (100 * p.need) / total)}%;` +
                                `border:1px solid ${p.need > p.usable ? "#e66" : "#ccc"};box-sizing:border-box;` }));
  root.append(bar, el("div", { class: "rte-quiet rte-line" },
    `in VRAM ${gb(used)}  ·  kept ${gb(p.mb)} + reserve ${gb(d.reserve_mb ?? 0)} of ${gb(d.usable_mb ?? 0)} usable`));
  if (d.scan && d.scan.done < d.scan.total)
    root.append(el("div", { class: "rte-quiet rte-line", title: d.scan.now ?? "" }, scanLine(d.scan)));

  const table = el("table");
  table.append(el("tr", {}, el("th", {}, local ? "keep" : "kept"), el("th", {}, "file"), el("th", {}, "type"),
                  el("th", {}, "size"),
                  el("th", { title: "text: Generate Text · vision: Remote Locate; ? = not checked yet" }, "can"),
                  el("th", {}, "state")));
  for (const m of d.models ?? []) {
    const row = p.rows[m.file] ?? { keep: false, type: "" };
    const edit = () => {
      state.draft ??= rowsFromServer(d);
      return state.draft[m.file];
    };
    let box, select; // another PC: a mark and the type as text, nothing to change
    if (local) {
      box = el("input", { type: "checkbox", checked: row.keep, disabled: !d.keep_file,
                          onchange: (e) => { edit().keep = e.target.checked; render(node); } });
      const types = (m.types ?? []).length ? [...m.types] : [...(d.all_types ?? [])];
      if (row.type && !types.includes(row.type)) types.unshift(row.type);
      if (!row.type) types.unshift(""); // no suggestion: the user picks, nothing is preselected
      select = el("select", { disabled: !d.keep_file,
                              title: "this file's type on this PC: kept as it, checked as it, and preselected in "
                                     + "Remote CLIP Loader (a loader node can still ask for another)",
                              onchange: (e) => { edit().type = e.target.value; render(node); },
                              onblur: () => setTimeout(() => { if (!editing(state)) render(node); }, 0) },
                  ...types.map((t) => el("option", { value: t, selected: t === row.type }, t || "pick a type")));
    } else {
      box = row.keep ? el("span", { title: "kept loaded on that PC" }, "\u2713") : null;
      select = row.type;
    }
    const loaded = m.loaded ?? [];
    let status;
    if ((m.kept_as ?? []).length) status = "kept";
    else if (loaded.length) status = `loaded (${loaded.map((l) => l.type).join(", ")})`;
    else if (!row.keep && p.need + m.size_mb > p.usable) status = "won't fit beside kept";
    else status = "not loaded";
    table.append(el("tr", {}, el("td", {}, box), el("td", { class: "rte-file" }, m.file), el("td", {}, select),
                    el("td", {}, gb(m.size_mb)), capabilityCell(m, row.type, d.scan),
                    el("td", { class: "rte-quiet" }, status)));
  }
  root.append(table);
  for (const err of d.keep_errors ?? []) root.append(el("div", { class: "rte-error" }, err));
  if (state.message) root.append(el("div", { class: state.messageError ? "rte-error" : "rte-quiet" }, state.message));

  if (!local) {
    root.append(el("div", { class: "rte-quiet" },
      "Read-only: the kept models are changed in that PC's ComfyUI (Remote Encoder Status with server_url local)."));
    return;
  }
  const over = p.need > p.usable;
  const untyped = (d.models ?? []).filter((m) => p.rows[m.file]?.keep && !p.rows[m.file].type).map((m) => m.file);
  const apply = el("button", { disabled: !state.draft || over || untyped.length || state.applying || !d.keep_file,
                               onclick: () => applyKeep(node) }, state.applying ? "Applying ..." : "Apply");
  const reset = el("button", { disabled: !state.draft || state.applying,
                               onclick: () => { state.draft = null; state.message = null; render(node); } }, "Reset");
  const note = !d.keep_file ? "this server has no keep list (started without --keep-file)"
             : over ? `too much: ${gb(p.need)} needed, ${gb(p.usable)} usable`
             : untyped.length ? `pick a type for ${untyped.join(", ")}`
             : state.draft ? "unapplied changes" : "";
  root.append(el("div", { class: "rte-foot" }, apply, reset,
                  el("span", { class: over || untyped.length ? "rte-error" : "rte-quiet" }, note)));
}

async function refresh(node) {
  const state = node.__rteStatus;
  const url = widget(node, "server_url")?.value;
  if (!url || state.applying || state.polling) return; // one request at a time: a slow reply must not pile up
  state.polling = true;
  try {
    const data = await (await api.fetchApi(`/remote_te/relay/status?url=${encodeURIComponent(url)}`)).json();
    if (widget(node, "server_url")?.value !== url) return; // changed meanwhile
    state.notice = null;
    if (data?.starting) {
      state.error = null;
      state.notice = data.error; // the local server is starting: not a failure
    } else if (data?.error) {
      state.error = data.error;
    } else if (!Array.isArray(data?.models)) {
      state.error = `${url} is not a remote text encoder server (unexpected /status reply)`;
    } else {
      state.error = null;
      state.data = data;
      state.versionWarning = isLocal(url) ? null : await versionWarning(url, data.version);
    }
  } catch (e) {
    state.error = `${url}: ${e}`;
  } finally {
    state.polling = false;
  }
  if (!editing(state)) render(node);
}

async function applyKeep(node) {
  const state = node.__rteStatus;
  const url = widget(node, "server_url")?.value;
  const rows = state.draft;
  if (!rows || !state.data || !isLocal(url)) return;
  const keep = (state.data.models ?? []).filter((m) => rows[m.file]?.keep)
    .map((m) => ({ file: m.file, type: rows[m.file].type }));
  const server = rowsFromServer(state.data);
  const types = Object.fromEntries(Object.entries(rows)
    .filter(([file, r]) => r.type && r.type !== server[file]?.type).map(([file, r]) => [file, r.type]));
  state.applying = true;
  state.message = `loading ${keep.length} kept model(s) on the server ...`;
  state.messageError = false;
  render(node);
  try {
    const res = await api.fetchApi(`/remote_te/relay/keep?url=${encodeURIComponent(url)}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ keep, types }),
    });
    const data = await res.json();
    if (res.ok && Array.isArray(data?.models)) {
      state.data = data;
      state.draft = null;
      const picks = Object.entries(types).filter(([file]) => !keep.some((k) => k.file === file));
      state.message = `kept: ${keep.length ? keep.map((k) => k.file).join(", ") : "nothing"}` +
        (picks.length ? `; type saved: ${picks.map(([f, t]) => `${f} as ${t}`).join(", ")}` : "");
    } else {
      state.message = data?.error ?? `Apply failed (${res.status})`;
      state.messageError = true;
    }
  } catch (e) {
    state.message = `Apply failed: ${e}`;
    state.messageError = true;
  } finally {
    state.applying = false;
  }
  await refresh(node);
}

function setup(node) {
  if (node.__rteStatus) return;
  addStyle();
  const root = el("div", { class: "rte-status" });
  const state = { root, data: null, draft: null, error: null, message: null, applying: false, timer: null };
  node.__rteStatus = state;
  const w = node.addDOMWidget("status", "remote_te_status", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => MIN_HEIGHT,
  });
  w.serialize = false;
  const url = widget(node, "server_url");
  if (url) {
    const original = url.callback;
    let t;
    url.callback = function (...args) {
      const r = original?.apply(this, args);
      clearTimeout(t);
      t = setTimeout(() => {
        state.data = null; state.draft = null; state.message = null;
        refresh(node);
      }, 500);
      return r;
    };
  }
  state.timer = setInterval(() => {
    if (!node.graph) return clearInterval(state.timer); // removed
    if (!document.hidden) refresh(node);
  }, POLL_MS);
  const removed = node.onRemoved;
  node.onRemoved = function (...args) {
    clearInterval(state.timer);
    return removed?.apply(this, args);
  };
  if ((node.size?.[0] ?? 0) < 520) node.setSize?.([520, Math.max(node.size?.[1] ?? 0, 360)]);
  render(node);
  refresh(node);
}

function useDefaultUrl(node) {
  const url = node.widgets?.find((w) => w.name === "server_url");
  const value = defaultServerUrl();
  if (url && value) url.value = value;
}

app.registerExtension({
  name: "remote_te.status",
  nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    useDefaultUrl(node); // a node loaded from a workflow gets its saved address right after this
    setup(node);
  },
  loadedGraphNode(node) {
    if (node.comfyClass !== NODE) return;
    setup(node);
    const state = node.__rteStatus; // the saved server_url is set now: ask that server
    state.data = null;
    state.draft = null;
    refresh(node);
  },
});
