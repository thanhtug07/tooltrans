import { useCallback, useEffect, useMemo, useState } from "react";

import { onModelDownloadProgress } from "@/api/events";
import {
  downloadModel,
  listLocalModels,
  modelCatalog,
  type LocalModelInfo,
  type ModelCatalogEntry,
} from "@/api/models";
import {
  createProvider,
  deleteProvider,
  setProviderDefault,
  setProviderEnabled,
  testProvider,
  updateProvider,
  type ProviderCapability,
  type ProviderInput,
  type ProviderKind,
  type ProviderView,
} from "@/api/provider";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/toast";
import { useProviders } from "@/stores/providers";

const CAPABILITY_LABELS: Record<ProviderCapability, string> = {
  translation: "Translation",
  stt: "STT",
  tts: "TTS",
};

const INPUT_CLS = "w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm";
const LABEL_CLS = "text-xs font-medium text-muted-foreground";

function capLabel(cap: string): string {
  return CAPABILITY_LABELS[cap as ProviderCapability] ?? cap;
}

// ---- form state ------------------------------------------------------------

type FormState = {
  name: string;
  provider_kind: ProviderKind;
  capabilities: ProviderCapability[];
  base_url: string;
  model: string;
  config: string;
  api_key: string;
  test: boolean;
};

const EMPTY_FORM: FormState = {
  name: "",
  provider_kind: "gemini",
  capabilities: ["translation"],
  base_url: "",
  model: "",
  config: "{}",
  api_key: "",
  test: false,
};

function formFromProvider(p: ProviderView): FormState {
  return {
    name: p.name,
    provider_kind: p.provider_kind,
    capabilities: p.capabilities,
    base_url: p.base_url ?? "",
    model: p.model ?? "",
    config: JSON.stringify(p.config ?? {}, null, 2),
    api_key: "",
    test: false,
  };
}

function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="max-h-[90vh] w-full max-w-lg space-y-4 overflow-y-auto rounded-lg border border-border bg-card p-5">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold">{title}</h3>
          <button
            type="button"
            aria-label="Close"
            className="rounded-md px-2 py-1 text-muted-foreground hover:bg-accent"
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ---- main panel ------------------------------------------------------------

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  return `${bytes} B`;
}

/**
 * Local LLM model management: downloads a translation GGUF into the app-data
 * models dir (via the worker) with live progress, then can point the FREE
 * provider's `model_path` at it — so offline translation works without a
 * manually-started llama-server.
 */
export function LocalModelManager() {
  const toast = useToast();
  const { providers, refresh } = useProviders();
  const [catalog, setCatalog] = useState<ModelCatalogEntry[]>([]);
  const [installed, setInstalled] = useState<LocalModelInfo[]>([]);
  const [downloading, setDownloading] = useState<ModelCatalogEntry | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; message: string | null }>({
    fraction: 0,
    message: null,
  });

  const load = useCallback(async () => {
    try {
      const [cat, local] = await Promise.all([modelCatalog(), listLocalModels()]);
      setCatalog(cat.models);
      setInstalled(Array.isArray(local) ? local : []);
    } catch (error) {
      toast.push(`Cannot load the model catalog: ${String(error)}`, "error");
    }
  }, [toast]);

  useEffect(() => {
    void load();
    let disposed = false;
    let unlisten: (() => void) | null = null;
    void onModelDownloadProgress((event) => {
      setProgress(() => ({ fraction: event.progress, message: event.message ?? null }));
    }).then((dispose) => {
      if (disposed) dispose();
      else unlisten = dispose;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [load]);

  const handleDownload = useCallback(
    async (entry: ModelCatalogEntry) => {
      setDownloading(entry);
      setProgress({ fraction: 0, message: "Starting download…" });
      try {
        await downloadModel(entry.repo_id, entry.filename);
        toast.push(`Model "${entry.name}" downloaded`, "success");
        await load();
      } catch (error) {
        toast.push(`Download failed: ${String(error)}`, "error");
      } finally {
        setDownloading(null);
      }
    },
    [load, toast],
  );

  const handleUseAsFreeModel = useCallback(
    async (model: LocalModelInfo) => {
      const free = providers.find((p) => p.id === "free");
      if (!free) {
        toast.push("The built-in FREE provider is not configured.", "error");
        return;
      }
      try {
        await updateProvider(free.id, {
          name: free.name,
          provider_type: free.provider_type,
          provider_kind: "free",
          capabilities: free.capabilities,
          config: { ...(free.config ?? {}), model_path: model.path },
        });
        toast.push("FREE provider now uses this model file (offline LLM)", "success");
        await refresh();
      } catch (error) {
        toast.push(`Could not set the model: ${String(error)}`, "error");
      }
    },
    [providers, refresh, toast],
  );

  const installedFiles = useMemo(() => new Set(installed.map((m) => m.file_name)), [installed]);

  return (
    <div className="space-y-2 rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Local LLM model (free / offline)
        </p>
        <Button size="sm" variant="ghost" onClick={() => void load()}>
          Refresh
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Download a GGUF once; it is stored in the app data dir and needs no API key. Point the FREE
        provider at it for fully offline translation.
      </p>

      {catalog.length === 0 && <p className="text-xs text-muted-foreground">Loading catalog…</p>}

      <ul className="space-y-2">
        {catalog.map((entry) => {
          const alreadyInstalled = installedFiles.has(entry.filename);
          return (
            <li
              key={entry.id}
              data-role={`model-catalog-${entry.id}`}
              className="flex flex-wrap items-center justify-between gap-2"
            >
              <div>
                <p className="text-sm font-medium">{entry.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatBytes(entry.size_bytes)} · {entry.filename}
                </p>
              </div>
              {alreadyInstalled ? (
                <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-400">
                  Installed
                </span>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={downloading !== null}
                  data-role={`model-download-${entry.id}`}
                  onClick={() => void handleDownload(entry)}
                >
                  {downloading?.id === entry.id ? "Downloading…" : "Download model"}
                </Button>
              )}
            </li>
          );
        })}
      </ul>

      {downloading && (
        <div className="space-y-1">
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all"
              style={{ width: `${(progress.fraction * 100).toFixed(0)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              {(progress.fraction * 100).toFixed(1)}% · {downloading.name}
            </span>
            {progress.message && <span className="truncate pl-2">{progress.message}</span>}
          </div>
        </div>
      )}

      {installed.length > 0 && (
        <div className="space-y-1 pt-1">
          <p className="text-xs font-medium text-muted-foreground">Installed models</p>
          {installed.map((model) => (
            <div
              key={model.path}
              data-role="installed-model"
              className="flex flex-wrap items-center justify-between gap-2"
            >
              <span className="max-w-[60%] truncate text-xs" title={model.path}>
                {model.file_name} ({formatBytes(model.size_bytes)})
              </span>
              <Button size="sm" variant="outline" onClick={() => void handleUseAsFreeModel(model)}>
                Use for FREE provider
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ProvidersPanel() {
  const toast = useToast();
  const { providers, defaults, providersFor, defaultFor, refresh } = useProviders();
  const [form, setForm] = useState<FormState | null>(null);
  const [editing, setEditing] = useState<ProviderView | null>(null);
  const [deleting, setDeleting] = useState<ProviderView | null>(null);
  const [busy, setBusy] = useState(false);
  const [testingId, setTestingId] = useState<string | null>(null);

  const translationOptions = useMemo(() => providersFor("translation"), [providersFor, providers]);
  const translationDefault = defaultFor("translation");
  const saveBusy = busy && form !== null;

  const apply = useCallback(
    async (input: ProviderInput, runTest: boolean) => {
      setBusy(true);
      try {
        if (editing) {
          await updateProvider(editing.id, input, runTest);
          toast.push(runTest ? "Provider updated and test passed" : "Provider updated", "success");
        } else {
          await createProvider(input, runTest);
          toast.push(runTest ? "Provider created and test passed" : "Provider created", "success");
        }
        setForm(null);
        setEditing(null);
        await refresh();
      } catch (error) {
        toast.push(String(error), "error");
      } finally {
        setBusy(false);
      }
    },
    [editing, refresh, toast],
  );

  const handleSubmit = useCallback(
    (runTest: boolean) => {
      if (!form) return;
      let config: Record<string, unknown>;
      try {
        config = JSON.parse(form.config) as Record<string, unknown>;
      } catch {
        toast.push("Configuration must be valid JSON", "error");
        return;
      }
      void apply(
        {
          name: form.name.trim(),
          provider_type: "translation",
          provider_kind: form.provider_kind,
          capabilities: form.capabilities,
          base_url: form.base_url.trim() || null,
          model: form.model.trim() || null,
          config,
          api_key: form.api_key.trim() || null,
        },
        runTest,
      );
    },
    [apply, form, toast],
  );

  const handleDelete = useCallback(async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await deleteProvider(deleting.id);
      toast.push(`Provider "${deleting.name}" deleted`, "success");
      setDeleting(null);
      await refresh();
    } catch (error) {
      toast.push(String(error), "error");
    } finally {
      setBusy(false);
    }
  }, [deleting, refresh, toast]);

  const handleTest = useCallback(
    async (p: ProviderView) => {
      setTestingId(p.id);
      try {
        const result = await testProvider(p.id);
        toast.push(
          result.ok
            ? `Test passed (${result.latency_ms} ms): ${result.detail}`
            : `Test failed: ${result.detail}`,
          result.ok ? "success" : "error",
        );
      } catch (error) {
        toast.push(`Test failed: ${String(error)}`, "error");
      } finally {
        setTestingId(null);
        await refresh();
      }
    },
    [refresh, toast],
  );

  const handleSetDefault = useCallback(
    async (id: string) => {
      try {
        await setProviderDefault(id, "translation");
        toast.push("Default translation provider updated", "success");
        await refresh();
      } catch (error) {
        toast.push(String(error), "error");
      }
    },
    [refresh, toast],
  );

  const handleToggle = useCallback(
    async (p: ProviderView) => {
      try {
        await setProviderEnabled(p.id, !p.enabled);
        await refresh();
      } catch (error) {
        toast.push(String(error), "error");
      }
    },
    [refresh, toast],
  );

  return (
    <div className="space-y-4">
      {/* Default translation provider — compact row */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs text-muted-foreground">Default</span>
        <select
          data-role="default-translation-provider"
          className="rounded-md border border-input bg-background px-2 py-1.5 text-sm"
          value={translationDefault?.id ?? "free"}
          onChange={(e) => void handleSetDefault(e.target.value)}
        >
          {translationOptions.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <details className="rounded-md border border-border px-3 py-2">
        <summary className="cursor-pointer text-xs text-muted-foreground">Local models</summary>
        <div className="mt-2">
          <LocalModelManager />
        </div>
      </details>

      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Providers
        </p>
        <Button
          data-role="add-provider"
          size="sm"
          onClick={() => {
            setEditing(null);
            setForm(EMPTY_FORM);
          }}
        >
          + Add
        </Button>
      </div>

      {/* Cards — Name / Status / Default; click opens configure */}
      <ul className="space-y-2">
        {providers.map((p) => {
          const isDefault = defaults.translation === p.id;
          const isConnected = p.enabled && (!p.needs_key || p.api_key_configured);
          const isKeyMissing = p.enabled && p.needs_key && !p.api_key_configured;
          return (
            <li
              key={p.id}
              data-role={`provider-card-${p.id}`}
              className="glass-card flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-card/60 p-3.5 shadow-2xs"
            >
              <button
                type="button"
                className="min-w-0 flex-1 text-left"
                onClick={() => {
                  setEditing(p);
                  setForm(formFromProvider(p));
                }}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-foreground">{p.name}</span>
                  {isDefault && (
                    <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-amber-400 ring-1 ring-amber-500/20">
                      Default
                    </span>
                  )}
                </div>
                <div className="mt-1 flex items-center gap-2 text-xs">
                  {isConnected ? (
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400">
                      <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />{" "}
                      Connected
                    </span>
                  ) : isKeyMissing ? (
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-400">
                      <span className="size-1.5 rounded-full bg-amber-400" /> Key Missing
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
                      <span className="size-1.5 rounded-full bg-muted-foreground/60" /> Disabled
                    </span>
                  )}
                </div>
              </button>

              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setEditing(p);
                  setForm(formFromProvider(p));
                }}
                className="hover:border-primary/50"
              >
                Configure
              </Button>
            </li>
          );
        })}
      </ul>

      {/* Add / edit modal */}
      {form && (
        <Modal
          title={editing ? `Configure ${editing.name}` : "Add provider"}
          onClose={() => {
            if (!busy) {
              setForm(null);
              setEditing(null);
            }
          }}
        >
          <div className="space-y-3">
            <div>
              <label className={LABEL_CLS} htmlFor="provider-name">
                Provider name
              </label>
              <input
                id="provider-name"
                data-role="provider-name"
                className={INPUT_CLS}
                value={form.name}
                placeholder="e.g. My Gemini"
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div>
              <label className={LABEL_CLS} htmlFor="provider-kind">
                Provider kind
              </label>
              <select
                id="provider-kind"
                data-role="provider-kind"
                className={INPUT_CLS}
                value={form.provider_kind}
                disabled={editing?.id === "free"}
                onChange={(e) =>
                  setForm({ ...form, provider_kind: e.target.value as ProviderKind })
                }
              >
                <option value="gemini">Gemini (cloud)</option>
                <option value="local">Local LLM (llama.cpp / OpenAI-compatible)</option>
                <option value="mock">Mock (offline test)</option>
              </select>
            </div>
            <div>
              <span className={LABEL_CLS}>Capabilities</span>
              <div className="flex flex-wrap gap-4 pt-1">
                {(Object.keys(CAPABILITY_LABELS) as ProviderCapability[]).map((cap) => (
                  <label key={cap} className="flex items-center gap-1.5 text-sm">
                    <input
                      type="checkbox"
                      data-role={`capability-${cap}`}
                      checked={form.capabilities.includes(cap)}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          capabilities: e.target.checked
                            ? [...form.capabilities, cap]
                            : form.capabilities.filter((c) => c !== cap),
                        })
                      }
                    />
                    {capLabel(cap)}
                    {cap === "translation" && (
                      <span className="text-[10px] text-muted-foreground">(live)</span>
                    )}
                  </label>
                ))}
              </div>
              <p className="pt-1 text-[10px] text-muted-foreground">
                STT / TTS capabilities are stored for future builds; only Translation is active now.
              </p>
            </div>
            {form.provider_kind !== "mock" && (
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <label className={LABEL_CLS} htmlFor="provider-base-url">
                    Base URL (local server or API endpoint)
                  </label>
                  <input
                    id="provider-base-url"
                    data-role="provider-base-url"
                    className={INPUT_CLS}
                    placeholder={
                      form.provider_kind === "gemini" ? "(Gemini default)" : "http://127.0.0.1:8080"
                    }
                    value={form.base_url}
                    onChange={(e) => setForm({ ...form, base_url: e.target.value })}
                  />
                </div>
                <div>
                  <label className={LABEL_CLS} htmlFor="provider-model">
                    Model
                  </label>
                  <input
                    id="provider-model"
                    data-role="provider-model"
                    className={INPUT_CLS}
                    placeholder="gemini-flash-lite-latest"
                    value={form.model}
                    onChange={(e) => setForm({ ...form, model: e.target.value })}
                  />
                </div>
              </div>
            )}
            {form.provider_kind === "gemini" && (
              <div>
                <label className={LABEL_CLS} htmlFor="provider-api-key">
                  API key
                </label>
                <input
                  id="provider-api-key"
                  data-role="provider-api-key"
                  type="password"
                  className={INPUT_CLS}
                  placeholder={
                    editing?.api_key_configured
                      ? "•••••••• (stored) — leave empty to keep"
                      : "Paste key…"
                  }
                  value={form.api_key}
                  onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                />
                <p className="pt-1 text-[10px] text-muted-foreground">
                  Stored in the OS credential vault (Windows Credential Manager) — never in the
                  database, never shown back. With “Save &amp; Test” the key is stored only if the
                  test passes.
                </p>
              </div>
            )}
            <div>
              <label className={LABEL_CLS} htmlFor="provider-config">
                Additional configuration (JSON)
              </label>
              <textarea
                id="provider-config"
                data-role="provider-config"
                className={`${INPUT_CLS} font-mono`}
                rows={3}
                value={form.config}
                onChange={(e) => setForm({ ...form, config: e.target.value })}
              />
              <p className="pt-1 text-[10px] text-muted-foreground">
                e.g. {"{"}"model_path": "C:/models/q4.gguf"{"}"} for a local model file.
              </p>
            </div>
            <div className="flex flex-wrap justify-end gap-2 pt-1">
              {editing && editing.id !== "free" && (
                <>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={testingId === editing.id}
                    onClick={() => void handleTest(editing)}
                  >
                    {testingId === editing.id ? "Testing…" : "Test"}
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => void handleToggle(editing)}>
                    {editing.enabled ? "Disable" : "Enable"}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive"
                    onClick={() => {
                      setDeleting(editing);
                      setForm(null);
                    }}
                  >
                    Delete
                  </Button>
                </>
              )}
              <Button
                variant="ghost"
                disabled={busy}
                onClick={() => {
                  setForm(null);
                  setEditing(null);
                }}
              >
                Cancel
              </Button>
              <Button
                data-role="provider-save-test"
                variant="outline"
                disabled={saveBusy || form.name.trim().length === 0}
                onClick={() => handleSubmit(true)}
              >
                {saveBusy ? "Testing…" : "Save & Test"}
              </Button>
              <Button
                data-role="provider-save"
                disabled={saveBusy || form.name.trim().length === 0}
                onClick={() => handleSubmit(false)}
              >
                {saveBusy ? "Saving…" : "Save"}
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* Delete confirmation */}
      {deleting && (
        <Modal title={`Delete "${deleting.name}"?`} onClose={() => !busy && setDeleting(null)}>
          <p className="text-sm text-muted-foreground">
            This provider may be used by automation profiles. If it is the default translation
            provider, the default falls back to FREE.
          </p>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="ghost" disabled={busy} onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button
              data-role="provider-delete-confirm"
              variant="default"
              disabled={busy}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => void handleDelete()}
            >
              {busy ? "Deleting…" : "Delete"}
            </Button>
          </div>
        </Modal>
      )}
    </div>
  );
}
