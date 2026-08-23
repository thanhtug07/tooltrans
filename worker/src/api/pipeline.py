"""Worker pipeline stage routes (RELEASE-P0).

The Rust JobService dispatches each stage to the worker over the authenticated
loopback HTTP API. These routes wrap the existing stage services
(``audio_service`` / ``stt_service`` / ``translation_service`` /
``subtitle_service`` / ``render_service``) with a thin, cancellable, validated
HTTP surface:

- ``POST /v1/audio/extract``   — WAV 16k mono extraction (+ cancel + progress log)
- ``POST /v1/stt/transcribe``  — (existing, extended with ``job_id`` cancel)
- ``POST /v1/translate``       — contextual translation via a named provider
- ``POST /v1/subtitle``        — cues + ASS/SRT generation from transcript+translation
- ``POST /v1/render``          — libass burn-in render (+ cancel + progress log)
- ``POST /v1/jobs/{job_id}/cancel`` — cancel an in-flight stage

Cancellation model: each request may carry a ``job_id``; the worker registers a
``CancellationToken`` for it for the duration of the call and the cancel
endpoint sets it, so long operations (STT / render) abort promptly. Tokens are
removed when the call finishes (success or failure).

Security: every route requires the bearer token; request bodies never contain
command lines; paths are validated by the services themselves; error responses
are the canonical ``{"error": {code, message, recoverable}}`` envelope and
never embed stack traces, tokens, or full command lines.
"""

from __future__ import annotations

import logging
import re
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import json as _json
import time as _time

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.routes import require_bearer
from src.api.schemas import Transcript, Translation, TranslationBlock
from src.core.job import CancelledError, CancellationToken
from src.services.providers.base import ProviderError

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_bearer)])

# ---------------------------------------------------------------------------
# Cancellation registry
# ---------------------------------------------------------------------------

_cancel_lock = threading.Lock()
_cancel_tokens: dict[str, CancellationToken] = {}


@contextmanager
def _cancel_scope(job_id: str | None):
    """Register a cancellation token for ``job_id`` for the call's lifetime.

    An already-registered token (e.g. pre-cancelled by ``cancel_job``) is
    reused so a job that was cancelled between stages cannot silently start
    the next stage.
    """
    if job_id:
        with _cancel_lock:
            token = _cancel_tokens.get(job_id)
    else:
        token = None
    if token is None:
        token = CancellationToken()
        if job_id:
            with _cancel_lock:
                _cancel_tokens[job_id] = token
    try:
        yield token
    finally:
        if job_id:
            with _cancel_lock:
                _cancel_tokens.pop(job_id, None)


def cancel_job(job_id: str) -> bool:
    """Request cancellation of an in-flight stage; returns whether it existed."""
    with _cancel_lock:
        token = _cancel_tokens.get(job_id)
    if token is None:
        return False
    token.cancel()
    return True


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AudioExtractRequest(BaseModel):
    video_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    job_id: str | None = None


class TranslateRequest(BaseModel):
    """One translation stage: transcript + provider selection + context.

    ``provider`` is the registered provider name (``mock``/``gemini``/``local``).
    ``provider_config`` carries provider-specific non-secret options (base URL,
    model name, model path); secrets (API keys) are carried in ``api_key`` only
    when the provider needs them, and are never logged.
    """

    transcript: Transcript
    project_id: str = Field(min_length=1)
    provider: str = "mock"
    target_language: str = Field(min_length=2, max_length=8)
    model: str = Field(default="gemini-flash-lite-latest", min_length=1)
    glossary_ver: str = "0"
    glossary: dict[str, str] | None = None
    characters: dict[str, str] | None = None
    rules: list[str] | None = None
    api_key: str | None = None
    provider_config: dict[str, str] | None = None
    job_id: str | None = None


class SubtitleRequest(BaseModel):
    transcript: Transcript
    translation: Translation
    project_id: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    language: str | None = None
    job_id: str | None = None


class ProviderTestRequest(BaseModel):
    """One-shot provider connectivity test (Provider Management).

    ``provider_kind`` is the worker registry kind (``free``/``gemini``/
    ``local``/``mock``); ``provider_config`` carries the non-secret options
    (base URL / model / model path). ``api_key`` is used ONLY for this call
    and is never stored or logged.
    """

    provider_kind: str = Field(min_length=1)
    provider_config: dict[str, str] | None = None
    api_key: str | None = None


class ModelCatalogEntryModel(BaseModel):
    """One downloadable translation model (Settings → Providers).

    ``repo_id``/``filename`` identify the Hugging Face GGUF; ``size_bytes`` is
    the pinned upstream size (used for progress anchoring and resume). The
    catalog is the single source of truth — Rust/frontend never hard-code it.
    """

    id: str
    name: str
    repo_id: str
    filename: str
    size_bytes: int
    vram_hint_mb: int


class ModelDownloadRequest(BaseModel):
    """``POST /v1/models/download`` — download a translation GGUF on demand.

    ``repo_id`` + ``filename`` are forwarded from the catalog; ``local_dir``
    is the directory (created if missing) the model downloads into; the final
    file is ``local_dir/<filename>`` with resume via ``<filename>.part``.
    ``mirror_url`` optionally overrides the CDN base (China mirror) — always
    validated against the fixed repo layout before use.
    """

    repo_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    local_dir: str = Field(min_length=1)
    mirror: str | None = None
    job_id: str | None = None


class WatermarkTextRequest(BaseModel):
    text: str = Field(min_length=1)
    position: str = "bottom-right"
    margin: int = 24
    x: int = 0
    y: int = 0
    font_size: int = 48
    color: str = "#FFFFFFFF"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    rotation: float = 0.0
    font: str | None = None
    font_file: str | None = None


class WatermarkImageRequest(BaseModel):
    image_path: str = Field(min_length=1)
    position: str = "bottom-right"
    margin: int = 24
    x: int = 0
    y: int = 0
    width: int = 0
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class WatermarkRequest(BaseModel):
    text: WatermarkTextRequest | None = None
    image: WatermarkImageRequest | None = None


class RenderSubtitleCue(BaseModel):
    """One edited cue used to rebuild the render-time ASS."""

    cue_number: int = Field(ge=1)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str = Field(min_length=1)


class RenderSubtitleStyle(BaseModel):
    """Subtitle position override from the preview (dragged custom position).

    ``custom_x``/``custom_y`` are fractions (0..1) of the video frame with the
    text center as the anchor — matching the drag interaction in the preview.
    """

    position: Literal["bottom_center", "top_center", "custom"] = "bottom_center"
    custom_x: float | None = Field(default=None, ge=0.0, le=1.0)
    custom_y: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Detected source language — picks the per-language ASS default style
    #: (font/size), so the render-time ASS matches the subtitle stage's output.
    language: str | None = None


class RenderRequest(BaseModel):
    video_path: str = Field(min_length=1)
    subtitle_path: str | None = None
    output_path: str = Field(min_length=1)
    encoder: str | None = None
    preset: str = "medium"
    crf: int = Field(default=18, ge=0, le=51)
    watermark: WatermarkRequest | None = None
    #: Optional full-duration voice track (``/v1/tts/synthesize`` output) to
    #: mix over the original audio (original ducked to ~45%).
    voice_track_path: str | None = None
    #: Optional pre-processed audio track (``/v1/audio/process`` output). When
    #: set, it replaces the video's original audio (and becomes the base the
    #: voice track mixes over when dubbing).
    audio_track_path: str | None = None
    #: Edited cues + position override. When ``subtitle_cues`` is present it
    #: takes precedence over ``subtitle_path``: the renderer rebuilds the ASS
    #: from these cues so text edits / deletions / the dragged position all
    #: appear in the burned-in output.
    subtitle_cues: list[RenderSubtitleCue] | None = None
    subtitle_style: RenderSubtitleStyle | None = None
    check_window: tuple[float, float] | None = None
    job_id: str | None = None


class LogoRegionRequest(BaseModel):
    """Marked logo rectangle (source-pixel coords) + optional time window."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    time_start: float | None = Field(default=None, ge=0.0)
    time_end: float | None = Field(default=None, ge=0.0)


class LogoRemoveRequest(BaseModel):
    video_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    region: LogoRegionRequest
    job_id: str | None = None


class LogoResult(BaseModel):
    output_path: str


class ChunkedAutomationRequest(BaseModel):
    """Run the chunked parallel pipeline (TASK_AUTOMATION_PINELINE).

    One worker call processes the whole video in 30 s logical chunks under
    bounded concurrency, merging the per-chunk STT/translation/TTS results
    into the project's cache artifacts (transcript/translation/subtitles/
    voice track) exactly where the later render stage expects them.
    """

    job_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    project_dir: str = Field(min_length=1)
    source_video: str = Field(min_length=1)
    source_audio: str = Field(min_length=1)
    target_language: str = Field(min_length=2, max_length=8)
    source_language: str | None = None
    provider: str = "mock"
    provider_config: dict[str, str] | None = None
    api_key: str | None = None
    model: str = "gemini-flash-lite-latest"
    glossary_ver: str = "0"
    glossary: dict[str, str] | None = None
    characters: dict[str, str] | None = None
    rules: list[str] | None = None
    dub: bool = False
    voice: str | None = None
    tts_engine: str = "edge"
    stt_model: str = "large-v3"
    stt_device: str = "auto"
    stt_mode: str = "auto"
    stt_batch_size: int = 2
    chunk_duration: float = 30.0
    overlap: float = 2.0
    max_concurrency: int = 4
    # Per-stage pool sizes for the streaming pipeline (optional; defaults derive
    # from ``max_concurrency``). Set by benchmarks, never hard-coded at call sites.
    stt_workers: int | None = None
    translate_workers: int | None = None
    tts_workers: int | None = None
    max_retries: int = 2
    duration_tolerance: float = 0.5


class ChunkedFinalizeRequest(BaseModel):
    """Phase 13–16: final validation + output verification + cleanup.

    Called by Rust after the chunked job's internal render produced
    ``output_path``. The temp tree is removed ONLY when both validation and
    verification pass; on failure everything is kept for debugging/retry.
    """

    job_id: str = Field(min_length=1)
    project_dir: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    source_duration: float = Field(gt=0.0)
    duration_tolerance: float = 0.5


class AudioProcessRequestModel(BaseModel):
    """Request for ``/v1/audio/process`` (one real ffmpeg mode)."""

    video_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    mode: Literal["vocal_removal", "normalize", "denoise"] = "vocal_removal"
    job_id: str | None = None


class AudioProcessResult(BaseModel):
    output_path: str
    mode: str


# ---------------------------------------------------------------------------
# Error envelope helper (canonical §25.3)
# ---------------------------------------------------------------------------


def _error(code: str, message: str, *, recoverable: bool = False, http: int = status.HTTP_422_UNPROCESSABLE_ENTITY) -> JSONResponse:
    return JSONResponse(
        status_code=http,
        content={"error": {"code": code, "message": message, "recoverable": recoverable}},
    )


# ---------------------------------------------------------------------------
# Provider factory (keeps the pipeline decoupled from concrete providers)
# ---------------------------------------------------------------------------


def build_translation_provider(name: str, config: dict[str, str] | None, api_key: str | None):
    """Resolve a translation provider by name (ADR §3.3 FROZEN)."""
    from src.services.providers.translation.gemini_provider import GeminiProvider  # noqa: PLC0415
    from src.services.providers.translation.local_llm_provider import LocalLLMProvider  # noqa: PLC0415
    from src.services.providers.translation.mock_provider import MockProvider  # noqa: PLC0415

    config = config or {}
    if name == "mock":
        return MockProvider()
    if name == "gemini":
        return GeminiProvider(api_key=api_key, model=config.get("model"))
    if name == "local":
        return LocalLLMProvider(
            model_path=config.get("model_path"),
            server_url=config.get("server_url"),
            model=config.get("model"),
        )
    if name == "free":
        # FREE = the first-class local/free provider: no API key, no cloud
        # egress. Translation requires a local LLM server (llama.cpp /
        # OpenAI-compatible) or a model file; otherwise the error is explicit
        # — never a silent fallback to a fake provider.
        server_url = config.get("server_url") or config.get("base_url")
        model_path = config.get("model_path")
        if not server_url and not model_path:
            raise ProviderError(
                "E_PROVIDER_UNAVAILABLE",
                "FREE translation needs a local LLM server (llama.cpp / OpenAI-compatible). "
                "Configure one in Settings → Providers, or choose a cloud provider.",
            )
        return LocalLLMProvider(
            model_path=model_path,
            server_url=server_url,
            model=config.get("model"),
        )
    raise ProviderError("E_PROVIDER_UNAVAILABLE", f"No translation provider named {name!r}.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/v1/audio/extract")
def audio_extract(request: AudioExtractRequest) -> JSONResponse:
    """Extract a 16k mono WAV from ``video_path`` into ``output_path``."""
    from src.services.audio_service import AudioExtractResult, extract_audio  # noqa: PLC0415 - lazy
    from src.core.ffmpeg import FFmpegError  # noqa: PLC0415

    try:
        with _cancel_scope(request.job_id) as cancel:
            result: AudioExtractResult = extract_audio(
                request.video_path,
                request.output_path,
                cancel=cancel,
                on_progress=lambda fraction: cancel.set_progress(fraction, "extract-audio"),
            )
    except CancelledError:
        return _error("E_CANCELLED", "Audio extraction was cancelled.", http=status.HTTP_409_CONFLICT)
    except FFmpegError as exc:
        return _error(exc.code, exc.message, recoverable=exc.code == "E_FFMPEG_NOT_FOUND")
    return JSONResponse(
        {
            "output_path": result.output_path,
            "duration_seconds": result.duration_seconds,
            "file_size_bytes": result.file_size_bytes,
        }
    )


@router.post("/v1/translate")
def translate(request: TranslateRequest) -> JSONResponse:
    """Translate ``request.transcript`` segments with the named provider."""
    from src.services.providers.base import SourceSegment  # noqa: PLC0415
    from src.services.quality_service import ProviderError as QProviderError  # noqa: PLC0415
    from src.services.translation_service import TranslationService  # noqa: PLC0415

    try:
        provider = build_translation_provider(
            request.provider, request.provider_config, request.api_key
        )
    except ProviderError as exc:
        return _error(exc.code, exc.message)

    segments = [
        SourceSegment(idx=s.idx, segment_id=s.id, text=s.text, speaker=s.speaker)
        for s in request.transcript.segments
    ]
    try:
        service = TranslationService()
        with _cancel_scope(request.job_id) as cancel:
            blocks: list[TranslationBlock] = service.translate_segments(
                segments,
                target_language=request.target_language,
                provider=provider,
                model=request.model,
                glossary_ver=request.glossary_ver,
                glossary=request.glossary,
                characters=request.characters,
                rules=request.rules,
                cancel=cancel,
                on_progress=lambda fraction: cancel.set_progress(
                    fraction, "translate", f"{(fraction * 100):.0f}% translated"
                ),
            )
    except (ProviderError, QProviderError) as exc:
        return _error(exc.code, exc.message)
    except CancelledError:
        return _error("E_CANCELLED", "Translation was cancelled.", http=status.HTTP_409_CONFLICT)
    return JSONResponse(
        Translation(
            schema_version=1,
            target_language=request.target_language,
            model=request.model,
            blocks=blocks,
        ).model_dump()
    )


@router.post("/v1/providers/test")
def provider_test(request: ProviderTestRequest) -> JSONResponse:
    """Probe whether a provider kind is reachable/configured (Provider test).

    - ``mock`` always answers (offline deterministic test).
    - ``gemini`` validates the key + model against the live API (auth errors
      surface as ``E_API_AUTH``, missing key as ``E_API_KEY_MISSING``).
    - ``local``/``free`` require a local llama-server URL or a model file and
      probe the server health.

    ``request.api_key`` is used only for this call — never stored, never
    logged.
    """
    import time  # noqa: PLC0415

    kind = request.provider_kind
    config = request.provider_config or {}
    started = time.monotonic()
    try:
        if kind == "mock":
            detail = "Mock provider is always available (offline test)."
        elif kind == "gemini":
            from src.services.providers.translation.gemini_provider import (  # noqa: PLC0415
                E_API_AUTH,
                E_API_ERROR,
                E_API_RATE_LIMIT,
                GeminiProvider,
                _AUTH_CODES,
                _rest_code,
            )

            api_key = request.api_key
            if not api_key:
                return _error(
                    "E_API_KEY_MISSING",
                    "Gemini needs an API key to test.",
                    http=status.HTTP_400_BAD_REQUEST,
                )
            provider = GeminiProvider(api_key=api_key, model=config.get("model"))
            client = provider._resolve_client()  # E_PROVIDER_UNAVAILABLE when SDK missing
            try:
                client.models.get(model=provider.model_name)
            except Exception as exc:  # noqa: BLE001 - classify every API failure
                code = _rest_code(exc)
                if code in _AUTH_CODES:
                    raise ProviderError(
                        E_API_AUTH, "Gemini authentication failed (invalid API key)."
                    ) from exc
                if code == 429:
                    raise ProviderError(E_API_RATE_LIMIT, "Gemini rate limit hit.") from exc
                raise ProviderError(
                    E_API_ERROR, f"Gemini request failed (HTTP {code})."
                ) from exc
            detail = f"Connected — model {provider.model_name} reachable."
        elif kind in ("local", "free"):
            from src.services.providers.translation.local_llm_provider import _health_ok  # noqa: PLC0415

            server_url = config.get("server_url") or config.get("base_url") or ""
            model_path = config.get("model_path") or ""
            if not server_url and not model_path:
                return _error(
                    "E_PROVIDER_UNAVAILABLE",
                    "FREE/local translation needs a local LLM server URL or a model path — "
                    "configure one in Settings → Providers.",
                    http=status.HTTP_400_BAD_REQUEST,
                )
            if server_url:
                if not _health_ok(server_url):
                    return _error(
                        "E_API_ERROR",
                        f"Cannot reach the local LLM server at {server_url}.",
                        http=status.HTTP_400_BAD_REQUEST,
                    )
                detail = f"Connected — local LLM server healthy at {server_url}."
            else:
                import os  # noqa: PLC0415

                if not os.path.isfile(model_path):
                    return _error(
                        "E_LOCAL_LLM_START",
                        f"Model file not found: {model_path}",
                        http=status.HTTP_400_BAD_REQUEST,
                    )
                detail = f"Model file present: {model_path}"
        else:
            return _error(
                "E_PROVIDER_UNAVAILABLE",
                f"No provider kind named {kind!r}.",
                http=status.HTTP_400_BAD_REQUEST,
            )
    except ProviderError as exc:
        return _error(exc.code, exc.message, http=status.HTTP_400_BAD_REQUEST)
    latency_ms = int((time.monotonic() - started) * 1000)
    return JSONResponse({"ok": True, "latency_ms": latency_ms, "detail": detail})


@router.post("/v1/subtitle")
def subtitle(request: SubtitleRequest) -> JSONResponse:
    """Generate cues + ASS/SRT from a transcript + translation into ``output_dir``."""
    from src.services.subtitle_service import SubtitleError, SubtitleService  # noqa: PLC0415 - lazy

    try:
        with _cancel_scope(request.job_id) as cancel:
            doc = SubtitleService().from_transcript_and_translation(
                request.transcript,
                request.translation,
                language=request.language,
                output_dir=request.output_dir,
            )
    except SubtitleError as exc:
        return _error(exc.code, exc.message)
    except CancelledError:
        return _error("E_CANCELLED", "Subtitle generation was cancelled.", http=status.HTTP_409_CONFLICT)
    return JSONResponse(
        {
            "cues": [c.model_dump() for c in doc.document.cues],
            "ass_path": doc.document.output.ass_path or "",
            "srt_path": doc.document.output.srt_path or "",
            "warnings": list(doc.warnings),
        }
    )


class TTSCueRequest(BaseModel):
    """One cue to speak: seconds + translated text (mirrors subtitle cues)."""

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str = Field(min_length=1)


class TTSRequest(BaseModel):
    """Request body for ``POST /v1/tts/synthesize`` (dubbing voice track)."""

    cues: list[TTSCueRequest] = Field(min_length=1)
    voice: str | None = None
    engine: str = "edge"  # ``edge`` (cloud, default) or ``piper`` (local)
    language: str | None = None
    duration_seconds: float = Field(gt=0.0)
    output_dir: str = Field(min_length=1)
    job_id: str | None = None


@router.post("/v1/tts/synthesize")
def tts_synthesize(request: TTSRequest) -> JSONResponse:
    """Synthesize the translated cues into a full-duration voice track.

    The output ``voice_track.wav`` (16-bit mono 44.1 kHz) is mixed over the
    original audio by the render stage when ``voice_track_path`` is passed to
    ``POST /v1/render``. Engines: ``edge`` (Microsoft neural voices, default)
    and ``piper`` (local).
    """
    from src.services.tts_service import (  # noqa: PLC0415 - lazy
        TTSError,
        TTSCue,
        synthesize_cues,
    )

    cues = [TTSCue(start=c.start, end=c.end, text=c.text) for c in request.cues]
    try:
        with _cancel_scope(request.job_id) as cancel:
            result = synthesize_cues(
                cues,
                voice=request.voice,
                engine=request.engine,
                language=request.language,
                duration_seconds=request.duration_seconds,
                output_dir=request.output_dir,
                cancel=cancel,
                on_progress=lambda fraction: cancel.set_progress(
                    fraction, "tts", f"segment {round(fraction * len(cues))}/{len(cues)}"
                ),
            )
    except CancelledError:
        return _error("E_CANCELLED", "TTS was cancelled.", http=status.HTTP_409_CONFLICT)
    except TTSError as exc:
        return _error(exc.code, exc.message)
    return JSONResponse(
        {
            "voice_track_path": result.voice_track_path,
            "meta_path": result.meta_path,
            "cue_count": len(cues),
            "engine_used": result.engine_used,
            "voice_used": result.voice_used,
        }
    )


@router.get("/v1/tts/voices")
def tts_voices() -> JSONResponse:
    """Available TTS voices per engine (single source of truth).

    Served by the worker (mirrors ``tts_service.EDGE_VOICES`` /
    ``PIPER_VOICES``) so voice names are never hard-coded in Rust or the
    frontend. ``default`` is the per-engine fallback voice.
    """
    from src.services.tts_service import (  # noqa: PLC0415 - lazy
        EDGE_VOICES,
        PIPER_VOICES,
        _DEFAULT_VOICE_FALLBACK,
        available_engines,
        voice_meta,
    )

    installed = set(available_engines())
    engines = []
    for engine_id, label, voices in (
        ("edge", "Edge (cloud — Microsoft neural, best quality)", EDGE_VOICES),
        ("piper", "Piper (local — offline, lower quality)", PIPER_VOICES),
    ):
        engines.append(
            {
                "id": engine_id,
                "label": label,
                "available": engine_id in installed,
                "voices": [
                    {"id": vid, "label": vlabel, **voice_meta(engine_id, vid)}
                    for vid, vlabel in voices.items()
                ],
            }
        )
    return JSONResponse(
        {
            "engines": engines,
            "defaults": {
                engine: {"voice": voice} for engine, voice in _DEFAULT_VOICE_FALLBACK.items()
            },
        }
    )


class TTSPreviewRequest(BaseModel):
    """Request body for ``POST /v1/tts/preview`` (Voice Library preview)."""

    engine: str = "edge"  # ``edge`` (cloud, default) or ``piper`` (local)
    voice: str = Field(min_length=1)
    text: str = Field(min_length=1)
    #: Cache directory (the Rust core passes the app-data voice-previews dir
    #: so the asset protocol can serve the file); default = system temp.
    output_dir: str | None = None


@router.post("/v1/tts/preview")
def tts_preview(request: TTSPreviewRequest) -> JSONResponse:
    """Synthesize one short clip for a voice preview (real TTS, never fake).

    Cached on disk by ``engine + voice + text`` hash so repeated previews of
    the same voice reuse the file instead of re-synthesizing.
    """
    import hashlib
    import tempfile

    from src.services.tts_service import (  # noqa: PLC0415 - lazy
        TTSError,
        synthesize_preview,
    )

    cache_dir = Path(request.output_dir) if request.output_dir else Path(tempfile.gettempdir()) / "aivideo-tts-preview"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{request.engine}-{request.voice}-{hashlib.sha1(request.text.encode('utf-8')).hexdigest()[:12]}"
    out = cache_dir / f"{key}.wav"
    if out.is_file():
        try:
            with wave.open(str(out), "rb") as w:
                duration = w.getnframes() / float(w.getframerate())
            return JSONResponse(
                {"path": str(out), "duration_seconds": round(duration, 3), "cached": True}
            )
        except Exception:  # noqa: BLE001 - corrupt cache file → regenerate
            out.unlink(missing_ok=True)
    try:
        duration = synthesize_preview(
            request.voice,
            engine=request.engine,
            text=request.text,
            out_wav=str(out),
        )
    except TTSError as exc:
        return _error(exc.code, exc.message)
    return JSONResponse(
        {"path": str(out), "duration_seconds": round(duration, 3), "cached": False}
    )


@router.post("/v1/render")
def render(request: RenderRequest) -> JSONResponse:
    """Burn subtitles into ``video_path`` with libass and validate the output."""
    from src.services.render_service import (  # noqa: PLC0415 - lazy
        RenderConfig,
        RenderError,
        RenderResult,
        ImageWatermark,
        TextWatermark,
        WatermarkConfig,
        build_render_ass,
        render as render_video,
    )

    # When edited cues are supplied, rebuild the render-time ASS from them
    # (text edits, deletions and the dragged position) instead of burning the
    # subtitle stage's regenerated ``subtitle.ass``.
    subtitle_ass_text = None
    if request.subtitle_cues is not None:
        style = request.subtitle_style or RenderSubtitleStyle()
        cues = [
            c for c in request.subtitle_cues if c.end > c.start and c.text.strip()
        ]
        subtitle_ass_text = build_render_ass(
            cues,
            language=style.language,
            position=style.position,
            custom_x=style.custom_x,
            custom_y=style.custom_y,
        )

    watermark = None
    if request.watermark is not None:
        text = None
        image = None
        if request.watermark.text is not None:
            t = request.watermark.text
            text = TextWatermark(
                text=t.text,
                position=t.position,
                margin=t.margin,
                x=t.x,
                y=t.y,
                font_size=t.font_size,
                color=t.color,
                opacity=t.opacity,
                rotation=t.rotation,
                font=t.font,
                font_file=t.font_file,
            )
        if request.watermark.image is not None:
            img = request.watermark.image
            image = ImageWatermark(
                image_path=img.image_path,
                position=img.position,
                margin=img.margin,
                x=img.x,
                y=img.y,
                width=img.width,
                opacity=img.opacity,
            )
        watermark = WatermarkConfig(text=text, image=image)

    config = RenderConfig(
        input_path=request.video_path,
        subtitle_path=None if request.subtitle_cues is not None else request.subtitle_path,
        subtitle_ass_text=subtitle_ass_text,
        output_path=request.output_path,
        video_encoder=request.encoder,
        video_preset=request.preset,
        video_crf=request.crf,
        watermark=watermark,
        voice_track_path=request.voice_track_path,
        audio_track_path=request.audio_track_path,
        check_window=request.check_window,
    )
    try:
        with _cancel_scope(request.job_id) as cancel:
            result: RenderResult = render_video(
                config,
                cancel=cancel,
                on_progress=lambda p: cancel.set_progress(
                    p.fraction, "render", f"{(p.fraction * 100):.0f}% encoded"
                ),
            )
    except CancelledError:
        return _error("E_CANCELLED", "Render was cancelled.", http=status.HTTP_409_CONFLICT)
    except RenderError as exc:
        return _error(exc.code, exc.message)
    return JSONResponse(
        {
            "output_path": result.output_path,
            "encoder_used": result.encoder_used,
            "duration_seconds": result.duration_seconds,
            "width": result.width,
            "height": result.height,
            "fps": list(result.fps),
            "audio_streams": result.audio_streams,
        }
    )


# ---------------------------------------------------------------------------
# Logo removal + audio processing (custom workflow steps)
# ---------------------------------------------------------------------------


@router.post("/v1/automation/chunked")
def chunked_automation(request: ChunkedAutomationRequest) -> JSONResponse:
    """Run the chunked parallel pipeline (Phase 1–10 of the chunked task)."""
    from src.services.chunk_service import (  # noqa: PLC0415 - lazy
        ChunkFailedError,
        run_chunked_pipeline,
    )

    try:
        with _cancel_scope(request.job_id) as cancel:
            manifest = run_chunked_pipeline(
                job_id=request.job_id,
                project_id=request.project_id,
                project_dir=request.project_dir,
                source_video=request.source_video,
                source_audio=request.source_audio,
                target_language=request.target_language,
                source_language=request.source_language,
                provider=request.provider,
                provider_config=request.provider_config,
                api_key=request.api_key,
                model=request.model,
                glossary_ver=request.glossary_ver,
                glossary=request.glossary,
                characters=request.characters,
                rules=request.rules,
                dub=request.dub,
                voice=request.voice,
                tts_engine=request.tts_engine,
                stt_model=request.stt_model,
                stt_device=request.stt_device,
                stt_mode=request.stt_mode,
                stt_batch_size=request.stt_batch_size,
                chunk_duration=request.chunk_duration,
                overlap=request.overlap,
                max_concurrency=request.max_concurrency,
                stt_workers=request.stt_workers,
                translate_workers=request.translate_workers,
                tts_workers=request.tts_workers,
                max_retries=request.max_retries,
                duration_tolerance=request.duration_tolerance,
                cancel=cancel,
                on_progress=lambda fraction, stage, message, **kw: cancel.set_progress(
                    fraction, stage, message, **kw
                ),
                on_event=lambda level, message: cancel.set_event(level, message),
            )
    except CancelledError:
        return _error("E_CANCELLED", "Chunked automation was cancelled.", http=status.HTTP_409_CONFLICT)
    except ChunkFailedError as exc:
        return _error(exc.code, exc.message, recoverable=True)
    except ProviderError as exc:
        return _error(exc.code, exc.message)
    return JSONResponse(manifest)


@router.post("/v1/automation/finalize")
def chunked_finalize(request: ChunkedFinalizeRequest) -> JSONResponse:
    """Final validation + output verification + cleanup for a chunked job."""
    import os  # noqa: PLC0415

    from src.services.chunk_service import (  # noqa: PLC0415 - lazy
        CleanupManager,
        final_validation,
        verify_output,
    )

    cleanup = CleanupManager(os.path.join(request.project_dir, "temp", request.job_id))
    cleanup.transition("validating")
    issues = final_validation(
        request.output_path,
        request.source_duration,
        tolerance=request.duration_tolerance,
    )
    if issues:
        cleanup.keep_temp()
        return JSONResponse(
            {
                "validation": "FAIL",
                "verification": "SKIPPED",
                "issues": issues,
                "cleanup": "kept",
            }
        )
    cleanup.transition("output_ready")
    v_issues = verify_output(request.output_path)
    if v_issues:
        cleanup.keep_temp()
        return JSONResponse(
            {"validation": "PASS", "verification": "FAIL", "issues": v_issues, "cleanup": "kept"}
        )
    cleanup.transition("output_verified")
    cleaned = cleanup.cleanup()
    return JSONResponse(
        {
            "validation": "PASS",
            "verification": "PASS",
            "cleanup": "done" if cleaned else "kept",
        }
    )


@router.post("/v1/logo/remove")
def logo_remove(request: LogoRemoveRequest) -> JSONResponse:
    """Remove a marked logo region with ffmpeg ``delogo`` (custom step)."""
    from src.services.logo_service import (  # noqa: PLC0415 - lazy
        LogoError,
        LogoRegion,
        remove_logo,
    )

    region = LogoRegion(
        x=request.region.x,
        y=request.region.y,
        width=request.region.width,
        height=request.region.height,
        time_start=request.region.time_start,
        time_end=request.region.time_end,
    )
    try:
        with _cancel_scope(request.job_id) as cancel:
            output = remove_logo(
                request.video_path,
                request.output_path,
                region,
                cancel=cancel,
                on_progress=lambda f: cancel.set_progress(f, "logo", f"{(f * 100):.0f}% removed"),
            )
    except CancelledError:
        return _error("E_CANCELLED", "Logo removal was cancelled.", http=status.HTTP_409_CONFLICT)
    except LogoError as exc:
        return _error(exc.code, exc.message)
    return JSONResponse({"output_path": output})


@router.post("/v1/audio/process")
def audio_process(request: AudioProcessRequestModel) -> JSONResponse:
    """Process the video's audio (vocal removal / normalize / denoise)."""
    from src.services.audio_process_service import (  # noqa: PLC0415 - lazy
        AudioError,
        process_audio,
    )

    try:
        with _cancel_scope(request.job_id) as cancel:
            output = process_audio(
                request.video_path,
                request.output_path,
                request.mode,
                cancel=cancel,
                on_progress=lambda f: cancel.set_progress(f, "audio", f"{(f * 100):.0f}% processed"),
            )
    except CancelledError:
        return _error("E_CANCELLED", "Audio processing was cancelled.", http=status.HTTP_409_CONFLICT)
    except AudioError as exc:
        return _error(exc.code, exc.message)
    return JSONResponse({"output_path": output, "mode": request.mode})


# ---------------------------------------------------------------------------
# Model management (Settings → Providers → "Download LLM model")
# ---------------------------------------------------------------------------

#: Downloadable translation models (GGUF). Single source of truth: the worker
#: serves this catalog so no model name is hard-coded in Rust or the frontend.
#: The Qwen2.5-3B Q4_K_M entry is the model verified to translate vi→zh end-to-end
#: on a CPU-only machine (llama.cpp, ~2007 MB, measured upstream size pinned here).
_MODEL_CATALOG: list[ModelCatalogEntryModel] = [
    ModelCatalogEntryModel(
        id="qwen2.5-3b-instruct",
        name="Qwen2.5-3B Instruct (Q4_K_M)",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        size_bytes=2104932768,
        vram_hint_mb=2560,
    )
]


@router.get("/v1/models/catalog")
def model_catalog() -> JSONResponse:
    """List downloadable translation models with pinned sizes (no secrets)."""
    return JSONResponse(
        {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "repo_id": m.repo_id,
                    "filename": m.filename,
                    "size_bytes": m.size_bytes,
                    "vram_hint_mb": m.vram_hint_mb,
                }
                for m in _MODEL_CATALOG
            ]
        }
    )


_DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "ai-video-localization" / "models"
_VENDOR_MODELS_DIR = Path(__file__).resolve().parents[3] / "vendor" / "models"


@router.get("/v1/models/list_local")
def list_local_models() -> JSONResponse:
    """List locally installed translation GGUF files."""
    seen: dict[str, dict] = {}
    for search_dir in (_DEFAULT_MODELS_DIR, _VENDOR_MODELS_DIR):
        if not search_dir.is_dir():
            continue
        for gguf in search_dir.glob("*.gguf"):
            if gguf.name not in seen:
                seen[gguf.name] = {
                    "file_name": gguf.name,
                    "path": str(gguf),
                    "size_bytes": gguf.stat().st_size,
                }
    return JSONResponse(list(seen.values()))


#: Allowed mirror base URLs (China mirror + upstream). The mirror may only
#: override the *host* of the canonical Hugging Face layout; the repo path and
#: filename are always taken from the catalog so a mirror can never redirect a
#: download to an arbitrary URL.
_MIRROR_ALLOWLIST = ("https://huggingface.co", "https://hf-mirror.com")

#: Characters that make a filename unsafe as a path component (traversal /
#: separator injection). Catalog filenames are plain ``<name>.gguf``.
_SAFE_MODEL_FILENAME = r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$"


@router.post("/v1/models/download")
def model_download(request: ModelDownloadRequest) -> JSONResponse:
    """Download a translation GGUF into ``local_dir`` (resume + cancel).

    Uses ``model_downloader.download_file`` (stdlib urllib, Range-resume,
    progress callback, cancellation) so an interrupted download continues from
    where it stopped. On success returns the final path + size so providers can
    point their ``model_path`` at it.

    The ``repo_id``/``filename`` must exist in the shipped catalog and the
    filename must be a plain ``*.gguf`` basename — arbitrary specifiers would
    allow path traversal on ``local_dir`` or an arbitrary HTTP fetch.
    """
    from src.services.model_downloader import (  # noqa: PLC0415 - lazy
        E_DISK_FULL,
        E_MODEL_DOWNLOAD,
        ModelDownloadError,
        download_file,
    )

    catalog_entry = next(
        (
            entry
            for entry in _MODEL_CATALOG
            if entry.repo_id == request.repo_id and entry.filename == request.filename
        ),
        None,
    )
    if catalog_entry is None:
        return _error(
            "E_MODEL_UNKNOWN",
            "Unknown model — the requested repo/filename is not in the catalog.",
        )
    if not re.fullmatch(_SAFE_MODEL_FILENAME, request.filename):
        return _error(
            "E_MODEL_UNKNOWN",
            "Unsupported model filename — must be a plain *.gguf basename.",
        )
    expected_size_bytes = catalog_entry.size_bytes

    mirror = request.mirror
    if mirror is not None:
        normalized_mirror = mirror.rstrip("/")
        if normalized_mirror not in _MIRROR_ALLOWLIST:
            return _error(
                "E_MODEL_UNKNOWN",
                "Unsupported download mirror.",
            )

    dest_dir = Path(request.local_dir) if request.local_dir.strip() else _DEFAULT_MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / request.filename

    if dest_path.is_file():
        actual = dest_path.stat().st_size
        if actual == expected_size_bytes:
            return JSONResponse(
                {"path": str(dest_path), "size_bytes": actual, "cached": True}
            )
        # Size mismatch: stale file — let download_file resume a fresh .part.
        dest_path.unlink()

    host = (request.mirror or "https://huggingface.co").rstrip("/")
    url = f"{host}/{request.repo_id}/resolve/main/{request.filename}"
    # Security: repo_id/filename are validated against the catalog above, and
    # the mirror against the allowlist, so this URL is always the canonical
    # Hugging Face layout — never an arbitrary fetch location.
    try:
        with _cancel_scope(request.job_id) as cancel:
            download_file(
                url,
                dest_path,
                expected_size_bytes=expected_size_bytes,
                cancel=cancel,
                on_progress=lambda downloaded, total: cancel.set_progress(
                    (downloaded / total) if total else 0.0,
                    "model-download",
                    f"{_fmt_mb(downloaded)} / {_fmt_mb(total)} MB",
                ),
            )
    except CancelledError:
        return _error("E_CANCELLED", "Model download was cancelled.", http=status.HTTP_409_CONFLICT)
    except ModelDownloadError as exc:
        if exc.code == E_DISK_FULL:
            return _error(exc.code, exc.message, recoverable=True)
        return _error(exc.code, exc.message, recoverable=True)
    except Exception:  # noqa: BLE001 - network/IO edge cases (S310 allowlisted)
        return _error(E_MODEL_DOWNLOAD, "Model download failed (network error).", recoverable=True)

    size = dest_path.stat().st_size
    return JSONResponse({"path": str(dest_path), "size_bytes": size, "cached": False})


def _fmt_mb(value: float) -> int:
    return int(value // (1024 * 1024))


@router.post("/v1/jobs/{job_id}/cancel")
def cancel(job_id: str) -> JSONResponse:
    """Cancel an in-flight stage for ``job_id`` (idempotent)."""
    existed = cancel_job(job_id)
    if not existed:
        return JSONResponse({"cancelled": False})
    return JSONResponse({"cancelled": True})


@router.get("/v1/progress/{job_id}")
def job_progress(job_id: str) -> JSONResponse:
    """Live stage progress for an in-flight ``job_id`` (polled by Rust).

    Returns ``progress: null`` when no stage for this job is currently
    registered — the caller treats that as "no progress available" and keeps
    its own stage anchors. Never exposes paths, tokens, or command lines.

    The ``events`` array contains all log events enqueued since the last poll
    (FIFO order).  The Rust poller drains this list on every poll so
    intermediate chunk-started / chunk-assembled lines are never lost even
    when multiple chunks fire events within a single poll interval.

    ``tasks`` is the v2 orchestrator extension (Phase 3): per-task progress
    for a pipeline job. Each entry is ``{task_id, task_type, progress, stage}``.
    For single-stage jobs or pipeline jobs whose tasks use per-task job_ids,
    this array is empty and the top-level progress remains the source of truth.
    """
    with _cancel_lock:
        token = _cancel_tokens.get(job_id)
        # Collect per-task tokens for pipeline jobs (tasks use job_id = task.id)
        task_entries: list[dict] = []
        if token is not None:
            # If this job_id looks like a pipeline job, gather its task children
            # Tasks share the same project but have distinct job_ids (task.id).
            # We scan for tokens whose job_id starts with f"{job_id}:".
            prefix = f"{job_id}:"
            for tid, ttok in _cancel_tokens.items():
                if tid.startswith(prefix):
                    tp, ts, _ = ttok.get_progress()
                    # Derive task_type from task_id suffix (e.g. job_1:translate -> translate)
                    ttype = tid.split(":", 1)[-1] if ":" in tid else tid
                    task_entries.append({
                        "task_id": tid,
                        "task_type": ttype,
                        "progress": tp,
                        "stage": ts,
                    })
    if token is None:
        return JSONResponse({"job_id": job_id, "progress": None, "stage": None, "message": None, "events": [], "tasks": task_entries})
    progress, stage, message = token.get_progress()
    events = token.drain_events()
    return JSONResponse({
        "job_id": job_id,
        "progress": progress,
        "stage": stage,
        "message": message,
        "events": [{"level": level, "message": msg} for level, msg in events],
        "tasks": task_entries,
    })


# ---------------------------------------------------------------------------
# SSE realtime event stream
# ---------------------------------------------------------------------------

_SSE_POLL_INTERVAL = 0.1  # seconds between polls for new events
_SSE_KEEPALIVE_INTERVAL = 15.0  # seconds between keepalive comments
_SSE_MAX_CONNECTIONS = 10  # max concurrent SSE streams per worker
_sse_semaphore = threading.Semaphore(_SSE_MAX_CONNECTIONS)


@router.get("/v1/events/stream/{job_id}")
def event_stream(job_id: str):
    """Server-Sent Events stream for a job's structured events.

    The generator polls ``CancellationToken.get_events_since(cursor)`` every
    ``_SSE_POLL_INTERVAL`` seconds and emits each new event as an SSE ``data:``
    line.  The stream ends when:

    1. The token is cleaned up (scope exit -> removed from ``_cancel_tokens``).
    2. The client disconnects.

    For unknown ``job_id`` (no token registered), returns a JSON fallback
    compatible with ``/v1/progress`` so the frontend can fall back to polling.
    """

    # Check if the job has a token; if not, return JSON fallback.
    with _cancel_lock:
        token = _cancel_tokens.get(job_id)

    if token is None:
        # Unknown job -- return progress-shaped JSON (frontend polls this).
        return JSONResponse({"job_id": job_id, "progress": None, "stage": None, "message": None, "events": [], "tasks": []})

    if not _sse_semaphore.acquire(blocking=False):
        return JSONResponse(
            {"error": {"code": "E_TOO_MANY_STREAMS", "message": "SSE connection limit reached", "recoverable": True}},
            status_code=429,
        )

    def _generate():  # noqa: E501 — generator must be nested to capture token
        cursor = 0
        last_keepalive = _time.monotonic()
        try:
          while True:
            # Always drain first — events may have arrived between the
            # last poll and the token being removed (scope exit).
            events = token.get_events_since(cursor)
            for evt in events:
                cursor = evt["event_id"]
                yield f"data: {_json.dumps(evt)}\n\n"

            # Now check if the token is still registered.
            with _cancel_lock:
                alive = job_id in _cancel_tokens
            if not alive:
                # Final drain: events emitted between our last drain and
                # the token removal must not be lost.
                events = token.get_events_since(cursor)
                for evt in events:
                    cursor = evt["event_id"]
                    yield f"data: {_json.dumps(evt)}\n\n"
                break

            # Keepalive comment (prevents proxy/client timeouts).
            now = _time.monotonic()
            if now - last_keepalive >= _SSE_KEEPALIVE_INTERVAL:
                yield ": keepalive\n\n"
                last_keepalive = now

            _time.sleep(_SSE_POLL_INTERVAL)
        finally:
            _sse_semaphore.release()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

