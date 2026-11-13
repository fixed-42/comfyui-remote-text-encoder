import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SERVE = "RemoteTE.Serve";
const GPU = "RemoteTE.ServeGpu";
const DEFAULT_URL = "RemoteTE.DefaultServerUrl";
const CATEGORY = "Remote Text Encoder";

export function defaultServerUrl() {
  const v = app.extensionManager?.setting?.get(DEFAULT_URL);
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

let gpus = [{ text: "auto", value: "auto" }];
let syncing = false; // set while the settings are written from the backend's state: no POST then

function settings() {
  return app.extensionManager?.setting;
}

function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t) t.add({ severity, summary, detail, life: severity === "error" ? 10000 : 4000 });
  else console[severity === "error" ? "error" : "log"](`[remote_te] ${summary}: ${detail}`);
}

async function show(state) {
  const s = settings();
  if (!s || !state) return;
  syncing = true;
  try {
    if (s.get(SERVE) !== state.serve) await s.set(SERVE, state.serve);
    if (s.get(GPU) !== state.gpu) await s.set(GPU, state.gpu);
  } finally {
    syncing = false;
  }
}

function describe(state) {
  if (!state.serve) return "Off: this ComfyUI does not serve text encoders.";
  if (state.error) return state.error;
  const on = state.gpu_name ? ` on ${state.gpu_name}` : "";
  return `Serving text encoders${on}: other PCs use http://<this PC>:<ComfyUI's port> as Remote CLIP Loader's ` +
         `server_url (ComfyUI must run with --listen).`;
}

async function change(body) {
  let state;
  try {
    const r = await api.fetchApi("/remote_te/serve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state = await r.json();
    if (!r.ok) {
      toast("error", "Remote Text Encoder", state.error ?? `HTTP ${r.status}`);
      await show(state.serve === undefined ? null : state); // back to what is in effect
      return;
    }
  } catch (e) {
    toast("error", "Remote Text Encoder", `could not change the setting: ${e}`);
    return;
  }
  toast(state.error ? "error" : "info", "Remote Text Encoder", describe(state));
}

app.registerExtension({
  name: "remote_te.settings",
  settings: [
    {
      id: DEFAULT_URL,
      category: [CATEGORY, "Nodes", "DefaultServerUrl"],
      name: "Default server_url for new nodes",
      tooltip:
        "What a new Remote CLIP Loader or Remote Encoder Status starts with: the encoder PC's ComfyUI " +
        "(http://pc-b:8188), or local for this ComfyUI's own encoder server. Empty: the nodes' own defaults " +
        "(loader http://127.0.0.1:8188, status local). Nodes already in a workflow keep their address.",
      type: "text",
      defaultValue: "",
    },
    {
      id: SERVE,
      category: [CATEGORY, "Serving", "Serve"],
      name: "Serve text encoders to other PCs",
      tooltip:
        "This PC encodes for others (PC-B): starts the encoder server now, and with ComfyUI from then on. " +
        "Other PCs reach it through this ComfyUI's address, so start ComfyUI with --listen. " +
        "REMOTE_TE_SERVE where ComfyUI was started decides instead.",
      type: "boolean",
      defaultValue: false,
      sortOrder: 2,
      onChange: (value, old) => {
        if (old === undefined || syncing) return; // registration, or written from the backend's state
        change({ serve: !!value });
      },
    },
    {
      id: GPU,
      category: [CATEGORY, "Serving", "GPU"],
      name: "Encoder GPU",
      tooltip:
        "The card the encoder server uses, by CUDA device number (as --cuda-device counts them) and name; ComfyUI's " +
        "own card is marked. It may share ComfyUI's card. auto: the first card ComfyUI doesn't render on, or the " +
        "only one. A change restarts the encoder server. REMOTE_TE_GPU where ComfyUI was started decides instead.",
      type: "combo",
      defaultValue: "auto",
      sortOrder: 1,
      options: () => gpus,
      onChange: (value, old) => {
        if (old === undefined || syncing) return;
        change({ gpu: String(value) });
      },
    },
  ],
  async setup() {
    try {
      const r = await api.fetchApi("/remote_te/serve");
      const state = await r.json();
      if (!r.ok) {
        console.warn(`[remote_te] serving settings: ${state.error ?? r.status}`);
        return;
      }
      if (Array.isArray(state.gpus) && state.gpus.length) gpus = state.gpus;
      await show(state);
    } catch (e) {
      console.warn(`[remote_te] serving settings: ${e}`);
    }
  },
});
