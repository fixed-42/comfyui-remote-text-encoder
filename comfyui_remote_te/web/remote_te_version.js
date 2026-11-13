import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let mine; // Promise of this ComfyUI's version, or null when it can't be read
const toasted = new Set();

export function localVersion() {
  mine ??= api.fetchApi("/remote_te/version").then((r) => (r.ok ? r.json() : null)).then((d) => d?.version ?? null)
    .catch(() => null);
  return mine;
}

export async function versionWarning(url, remote) {
  const own = await localVersion();
  if (!own || remote === own) return null;
  const theirs = remote ? `version ${remote}` : "an older version (it reports none)";
  return `${url} runs ${theirs} of the remote text encoder pack, this ComfyUI ${own}: install the same version on ` +
         `both PCs`;
}

export function toastOnce(url, remote, text) {
  const key = `${url} ${remote}`;
  if (toasted.has(key)) return;
  toasted.add(key);
  console.warn(`[remote_te] ${text}`);
  app.extensionManager?.toast?.add({ severity: "warn", summary: "Remote Text Encoder", detail: text, life: 10000 });
}
