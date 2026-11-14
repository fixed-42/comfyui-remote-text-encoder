import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { defaultServerUrl } from "./remote_te_settings.js";
import { toastOnce, versionWarning } from "./remote_te_version.js";

const NO_MODEL = "(no model list yet)";   // the loader's placeholder until the menu is filled
const LEGACY_DEFAULT = "(server default)"; // stage 1 value in saved workflows
const NODE = "RemoteCLIPLoader";

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function describe(url, data) {
  return data.mode === "on_demand" ? `${url} (on-demand server, ${data.gpu ?? "GPU"})` : `${url} (preloaded server)`;
}

const isUnset = (v) => [undefined, null, "", NO_MODEL, LEGACY_DEFAULT].includes(v);

function typeOptions(info, allTypes) {
  const offered = new Set(info.entries.flatMap((e) => e.types));
  return [...allTypes.filter((t) => offered.has(t)), ...[...offered].filter((t) => !allTypes.includes(t))];
}

const filesFor = (info, type) => info.entries.filter((e) => e.types.includes(type)).map((e) => e.name);

function defaultFile(info, type) {
  const fits = info.entries.filter((e) => e.types.includes(type));
  return (fits.find((e) => e.kept) ?? fits.find((e) => e.loaded) ?? fits[0])?.name;
}

function defaultType(info, options) {
  const e = info.entries.find((e) => e.kept) ?? info.entries.find((e) => e.loaded);
  const t = e?.kept?.[0] ?? e?.loaded?.[0];
  return options.includes(t) ? t : options[0];
}

function setMenus(node, typeChanged) {
  const type = widget(node, "type");
  const model = widget(node, "model");
  const info = node.__remoteTe;
  if (!type || !model || !info) return;
  const options = typeOptions(info, type.__allTypes);
  const fresh = isUnset(model.value);
  const chosen = info.entries.find((e) => e.name === model.value);
  if (type.value === "qwen_image" && chosen?.types.includes("text_gen") && !chosen.types.includes("qwen_image"))
    type.value = "text_gen"; // a workflow from before text_gen: Qwen3.5 (prompt enhancer) files were qwen_image
  if (fresh && options.length) type.value = defaultType(info, options);
  type.options.values = options.includes(type.value) || !type.value ? options : [...options, type.value];
  if (!type.options.values.length) type.options.values = [...type.__allTypes];
  const files = filesFor(info, type.value);
  if ((fresh || typeChanged) && !files.includes(model.value)) model.value = defaultFile(info, type.value) ?? NO_MODEL;
  if (!isUnset(model.value) && !files.includes(model.value)) {
    files.push(model.value);
    console.warn(`[remote_te] '${model.value}' is not ${info.mode === "on_demand" ? "a file" : "a model"} for type ` +
                 `'${type.value}' on ${info.where}; pick one from the menu`);
  }
  model.options.values = files.length ? files : [NO_MODEL];
  node.setDirtyCanvas?.(true, true);
}

function mark(m, onDemand) {
  if (onDemand) return m.kept_as?.length ? "kept" : m.loaded_as?.length ? "loaded" : "";
  return m.up === false ? "down" : "";
}

function label(marks, value) {
  const m = marks?.[value];
  return m ? `${value}  (${m})` : value;
}

async function refresh(node) {
  const url = widget(node, "server_url")?.value;
  const model = widget(node, "model");
  if (!url || !model) return;
  let data;
  try {
    data = await (await api.fetchApi(`/remote_te/relay/models?url=${encodeURIComponent(url)}`)).json();
  } catch (e) {
    console.warn(`[remote_te] ${url}: ${e}`);
    return;
  }
  if (data?.error) {
    console.warn(`[remote_te] ${data.error}`);
    return; // keep the current options
  }
  if (!Array.isArray(data?.models)) {
    console.warn(`[remote_te] ${url} is not a remote text encoder server (unexpected /models reply)`);
    return;
  }
  if (url.trim().toLowerCase() !== "local") {
    const warning = await versionWarning(url, data.version);
    if (warning) toastOnce(url, data.version, warning);
  }
  const onDemand = data.mode === "on_demand";
  node.__remoteTe = {
    mode: onDemand ? "on_demand" : "preloaded",
    where: describe(url, data),
    entries: data.models.map((m) => ({
      name: m.name,
      types: onDemand ? m.types ?? [] : m.type ? [m.type] : [],
      kept: m.kept_as?.length ? m.kept_as : null,
      loaded: m.loaded_as?.length ? m.loaded_as : null,
    })),
    marks: Object.fromEntries(data.models.map((m) => [m.name, mark(m, onDemand)]).filter(([, v]) => v)),
  };
  setMenus(node, false);
}

function fixOldOrder(node) {
  const type = widget(node, "type");
  const model = widget(node, "model");
  if (!type || !model || !type.__allTypes) return;
  const known = (v) => type.__allTypes.includes(v);
  if (!known(type.value) && (known(model.value) || isUnset(model.value))) {
    const file = type.value;
    type.value = isUnset(model.value) ? "" : model.value;
    model.value = file;
  }
}

function hook(node) {
  const url = widget(node, "server_url");
  if (!url || url.__remoteTe) return;
  url.__remoteTe = true;
  const model = widget(node, "model");
  const type = widget(node, "type");
  if (model) model.options.getOptionLabel = (v) => label(node.__remoteTe?.marks, v); // else plain names
  if (type) {
    type.__allTypes ??= [...type.options.values]; // CLIPLoader's types, as the node declares them
    const picked = type.callback;
    type.callback = function (...args) {
      const r = picked?.apply(this, args);
      setMenus(node, true); // the user picked another type: the files for it
      return r;
    };
  }
  const executed = node.onExecuted;
  node.onExecuted = function (output) {
    const r = executed?.apply(this, arguments);
    const show = (w, value) => {
      if (!value || !w || w.value === value) return;
      if (!w.options.values.includes(value)) w.options.values = [...w.options.values, value];
      w.value = value;
    };
    show(type, output?.remote_te_type?.[0]);
    show(model, output?.remote_te_model?.[0]);
    setMenus(node, false); // the file menu for the new type, without entries left from before
    node.setDirtyCanvas?.(true, true);
    return r;
  };
  const original = url.callback;
  let timer;
  url.callback = function (...args) {
    const r = original?.apply(this, args);
    clearTimeout(timer);
    timer = setTimeout(() => refresh(node), 500);
    return r;
  };
}

function useDefaultUrl(node) {
  const url = node.widgets?.find((w) => w.name === "server_url");
  const value = defaultServerUrl();
  if (url && value) url.value = value;
}

app.registerExtension({
  name: "remote_te.model_menu",
  async setup() {
    const all = () => {
      for (const n of app.graph?._nodes ?? []) if (n.comfyClass === NODE) refresh(n);
    };
    api.addEventListener("promptQueued", all); // after Run, not before: an unreachable server must not delay the run
    api.addEventListener("execution_success", all); // the run loaded models: the "loaded" marks changed
  },
  nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    useDefaultUrl(node); // a node loaded from a workflow gets its saved address right after this
    hook(node);
    setTimeout(() => refresh(node), 0);
  },
  loadedGraphNode(node) {
    if (node.comfyClass !== NODE) return;
    hook(node);
    fixOldOrder(node);
    refresh(node);
  },
  async refreshComboInNodes() {
    await Promise.all((app.graph?._nodes ?? []).filter((n) => n.comfyClass === NODE).map(refresh));
  },
});
