/**
 * Web API command invocation wrapper.
 * All IPC commands are routed through HTTP to the Python worker.
 */

const WORKER_BASE = "http://127.0.0.1:8765";
const AUTH_TOKEN = "dev-placeholder-token";

const HTTP_ROUTES: Record<
  string,
  {
    method: "GET" | "POST" | "DELETE" | "PUT";
    path: string;
    buildRequest: (args: Record<string, unknown>) => { params?: Record<string, string>; body?: unknown };
  }
> = {
  "ping": { method: "GET", path: "/api/health", buildRequest: () => ({}) },
  // Projects
  "project.list": { method: "GET", path: "/api/projects", buildRequest: () => ({}) },
  "project.findBySourceVideo": { method: "GET", path: "/api/projects/by-source", buildRequest: ({ videoPath }) => ({ params: { video_path: String(videoPath) } }) },
  "project.open": { method: "GET", path: "/api/projects", buildRequest: ({ id }) => ({ params: { _pathParam: String(id) } }) },
  "project.create": { method: "POST", path: "/api/projects", buildRequest: ({ name, videoPath }) => ({ body: { name: String(name), videoPath: String(videoPath) } }) },
  "project.save": { method: "POST", path: "/api/projects", buildRequest: ({ id }) => ({ params: { _pathParam: `${String(id)}/save` }, body: {} }) },
  "project.delete": { method: "DELETE", path: "/api/projects", buildRequest: ({ id }) => ({ params: { _pathParam: String(id) } }) },
  // Settings
  "settings.get_all": { method: "GET", path: "/api/settings", buildRequest: () => ({}) },
  "settings.set": { method: "POST", path: "/api/settings", buildRequest: ({ key, value }) => ({ body: { key: String(key), value: String(value) } }) },
  // TTS
  "settings.voices": { method: "GET", path: "/api/tts/voices", buildRequest: () => ({}) },
  "settings.ttsPreview": { method: "POST", path: "/api/tts/preview", buildRequest: ({ engine, voice, text }) => ({ body: { engine: String(engine), voice: String(voice), text: String(text) } }) },
  // Jobs
  "job.list": { method: "GET", path: "/api/jobs", buildRequest: () => ({}) },
  "job.list_all": { method: "GET", path: "/api/jobs", buildRequest: () => ({}) },
  "job.get": { method: "GET", path: "/api/jobs", buildRequest: ({ jobId, id }) => ({ params: { _pathParam: String(jobId || id) } }) },
  "job.submit": { method: "POST", path: "/api/jobs", buildRequest: (args) => ({ body: args }) },
  "job.cancel": { method: "POST", path: "/v1/jobs", buildRequest: ({ jobId, id }) => ({ params: { _pathParam: `${String(jobId || id)}/cancel` } }) },
  "job.retry": { method: "POST", path: "/v1/jobs", buildRequest: ({ jobId, id }) => ({ params: { _pathParam: `${String(jobId || id)}/retry` } }) },
  // Tasks
  "task.list": { method: "GET", path: "/api/tasks", buildRequest: ({ jobId }) => ({ params: jobId ? { job_id: String(jobId) } : undefined }) },
  "task.get": { method: "GET", path: "/api/tasks", buildRequest: ({ taskId, id }) => ({ params: { _pathParam: String(taskId || id) } }) },
  // Media
  "media.probe": { method: "GET", path: "/api/media/probe", buildRequest: ({ path }) => ({ params: { path: String(path) } }) },
  // Worker / system
  "worker.get_worker_state": { method: "GET", path: "/api/worker/state", buildRequest: () => ({}) },
  "worker.restart": { method: "POST", path: "/api/worker/state", buildRequest: () => ({ body: { action: "restart" } }) },
  "system.hardware": { method: "GET", path: "/api/system/hardware", buildRequest: () => ({}) },
  "system.reveal": { method: "POST", path: "/api/settings", buildRequest: ({ path }) => ({ body: { key: "system.reveal", value: String(path) } }) },
  // Models
  "models.catalog": { method: "GET", path: "/v1/models/catalog", buildRequest: () => ({}) },
  "models.list_local": { method: "GET", path: "/v1/models/list_local", buildRequest: () => ({}) },
  "models.download": { method: "POST", path: "/v1/models/download", buildRequest: ({ repoId, filename, mirror, localDir }) => ({ body: { repo_id: String(repoId), filename: String(filename), local_dir: String(localDir || ""), mirror: mirror ? String(mirror) : null } }) },
  // Providers
  "providers.list": { method: "GET", path: "/api/providers", buildRequest: () => ({}) },
  "providers.get": { method: "GET", path: "/api/providers", buildRequest: ({ id }) => ({ params: { _pathParam: String(id) } }) },
  "providers.create": { method: "POST", path: "/api/providers", buildRequest: ({ input }) => ({ body: input }) },
  "providers.update": { method: "PUT", path: "/api/providers", buildRequest: ({ id, input }) => ({ params: { _pathParam: String(id) }, body: input }) },
  "providers.delete": { method: "DELETE", path: "/api/providers", buildRequest: ({ id }) => ({ params: { _pathParam: String(id) } }) },
  "providers.test": { method: "POST", path: "/api/providers", buildRequest: ({ id }) => ({ params: { _pathParam: `${String(id)}/test` } }) },
  "providers.set_default": { method: "POST", path: "/api/providers", buildRequest: ({ id }) => ({ params: { _pathParam: `${String(id)}/default` } }) },
  "providers.set_enabled": { method: "POST", path: "/api/providers", buildRequest: ({ id, enabled }) => ({ params: { _pathParam: `${String(id)}/enable` }, body: { enabled } }) },
  // Secrets
  "secrets.get_api_key_masked": { method: "GET", path: "/api/settings", buildRequest: () => ({ params: { _secret: "1" } }) },
  "secrets.set_api_key": { method: "POST", path: "/api/settings", buildRequest: ({ provider, key }) => ({ body: { key: `api.key.${provider}`, value: String(key) } }) },
  "secrets.delete_api_key": { method: "POST", path: "/api/settings", buildRequest: ({ provider }) => ({ body: { key: `api.key.${provider}`, value: "" } }) },
  // Pipeline
  "pipeline.artifact_paths": { method: "GET", path: "/api/pipeline/artifact-paths", buildRequest: ({ projectId }) => ({ params: { project_id: String(projectId) } }) },
  "pipeline.submit": { method: "POST", path: "/api/pipeline/submit", buildRequest: (args) => ({ body: args }) },
  // Export
  "export.video": { method: "POST", path: "/v1/export/video", buildRequest: (args) => ({ body: args }) },
  "export.subtitles": { method: "POST", path: "/v1/export/subtitles", buildRequest: (args) => ({ body: args }) },
  // Subtitle cues
  "subtitle.get_cues": { method: "GET", path: "/api/subtitle/cues", buildRequest: ({ projectId }) => ({ params: { project_id: String(projectId) } }) },
  "subtitle.replace_cues": { method: "POST", path: "/api/subtitle/cues", buildRequest: ({ projectId, cues }) => ({ body: { projectId, cues } }) },
  "subtitle.update_cue": { method: "PUT", path: "/api/subtitle/cues", buildRequest: ({ cueId }) => ({ params: { _pathParam: String(cueId) } }) },
  // Dictionary
  "dictionary.character.list": { method: "GET", path: "/api/dictionary/characters", buildRequest: () => ({}) },
  "dictionary.character.upsert": { method: "POST", path: "/api/dictionary/characters", buildRequest: (args) => ({ body: args }) },
  "dictionary.character.delete": { method: "DELETE", path: "/api/dictionary/characters", buildRequest: ({ id }) => ({ params: { _pathParam: String(id) } }) },
  "dictionary.glossary.list": { method: "GET", path: "/api/dictionary/glossary", buildRequest: () => ({}) },
  "dictionary.glossary.upsert": { method: "POST", path: "/api/dictionary/glossary", buildRequest: (args) => ({ body: args }) },
  "dictionary.glossary.delete": { method: "DELETE", path: "/api/dictionary/glossary", buildRequest: ({ id }) => ({ params: { _pathParam: String(id) } }) },
  "dictionary.glossary.fingerprint": { method: "POST", path: "/api/dictionary/glossary", buildRequest: ({ id }) => ({ params: { _pathParam: `${String(id)}/fingerprint` } }) },
};

function buildUrl(base: string, path: string, params?: Record<string, string>, pathParam?: string): string {
  let url = pathParam ? `${base}${path}/${pathParam}` : `${base}${path}`;
  if (params) {
    const qs = Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== "")
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join("&");
    if (qs) url += `?${qs}`;
  }
  return url;
}

async function httpInvoke<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  const route = HTTP_ROUTES[cmd];
  if (!route) {
    throw new Error(`Command "${cmd}" is not available through the HTTP interface in web mode.`);
  }
  const { params, body } = route.buildRequest(args);
  // Extract _pathParam for path segments
  let pathParam: string | undefined;
  const cleanParams: Record<string, string> = {};
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (k === "_pathParam") pathParam = v as string;
      else cleanParams[k] = v as string;
    }
  }
  const url = buildUrl(WORKER_BASE, route.path, cleanParams, pathParam);
  const fetchOpts: RequestInit = { method: route.method };
  const headers: Record<string, string> = { "Authorization": `Bearer ${AUTH_TOKEN}` };
  if (route.method !== "GET") headers["Content-Type"] = "application/json";
  fetchOpts.headers = headers;
  if (body) fetchOpts.body = JSON.stringify(body);
  const res = await fetch(url, fetchOpts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText} — ${cmd}: ${text}`);
  }
  return (await res.json()) as T;
}

export async function safeInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  return httpInvoke<T>(cmd, args ?? {});
}
