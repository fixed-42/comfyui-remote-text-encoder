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

function setTypes(node, pick) {
  const type = widget(node, "type");
  const model = widget(node, "model");
  const info = node.__remoteTe;
  if (!type || !model || !info) return;
  type.__allTypes ??= [...type.options.values]; // CLIPLoader's types, as the node declares them
  const suggested = info.types[model.value] ?? [];
  if (suggested.length && (info.mode !== "on_demand" || pick)) type.value = suggested[0]; // preloaded: the server's
  const values = suggested.length ? [...suggested] : [...type.__allTypes];
  if (!values.includes(type.value)) values.push(type.value); // keep a saved type the server doesn't suggest
  type.options.values = values;
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
    types: Object.fromEntries(data.models.map((m) => [m.name, onDemand ? m.types ?? [] : m.type ? [m.type] : []])),
    marks: Object.fromEntries(data.models.map((m) => [m.name, mark(m, onDemand)]).filter(([, v]) => v)),
  };
  const names = data.models.map((m) => m.name);
  const values = [...names];
  const unset = [undefined, null, "", NO_MODEL, LEGACY_DEFAULT].includes(model.value);
  let pick = false;
  if (unset) {
    if (names.length) {
      model.value = names[0]; // like any ComfyUI menu: the first entry
      pick = true;
    }
  } else if (!values.includes(model.value)) {
    values.push(model.value); // keep a saved choice the server lacks; running it fails with the server's list
    console.warn(`[remote_te] '${model.value}' is not ${onDemand ? "a file" : "a model"} on ${describe(url, data)}; ` +
                 `pick one from the menu`);
  }
  model.options.values = values.length ? values : [NO_MODEL];
  setTypes(node, pick);
}

function hook(node) {
  const url = widget(node, "server_url");
  if (!url || url.__remoteTe) return;
  url.__remoteTe = true;
  const model = widget(node, "model");
  if (model) {
    model.options.getOptionLabel = (v) => label(node.__remoteTe?.marks, v); // frontends without it show plain names
    const picked = model.callback;
    model.callback = function (...args) {
      const r = picked?.apply(this, args);
      setTypes(node, true); // the user picked another model
      return r;
    };
  }
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
    refresh(node);
  },
  async refreshComboInNodes() {
    await Promise.all((app.graph?._nodes ?? []).filter((n) => n.comfyClass === NODE).map(refresh));
  },
});
