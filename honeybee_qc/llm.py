"""Model access: a small protocol, a Claude CLI adapter, and a fake.

Every stage depends on `ModelClient`, never on a subprocess, so the whole audit
runs in tests with no network, no key, and no cost.

The CLI is invoked with tools disabled. The predecessor engine granted Bash,
Read, Glob, and Grep and let the model explore a filesystem, which made runs slow
and non-reproducible. Here all evidence is stuffed into the prompt, so there is
nothing to explore and nothing to vary between runs.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Protocol

from .cache import ResponseCache, compute_cache_key

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_EFFORT = "medium"


@dataclass
class ModelRequest:
    key: str
    prompt: str
    schema: dict[str, Any]
    system: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Which independent draw of this judgment this request is. Two requests that
    # differ only here are the same question asked twice on purpose, so the cache
    # must treat them as distinct rows; see `compute_cache_key`. Zero is the
    # single-sample default and keys exactly as it did before sampling existed.
    sample_index: int = 0
    # Which model should answer this one request. Empty means the client's own,
    # which is every request unless something routed it; see `RoutedClient`. It
    # is part of the cache key wherever it is set, because an answer is only
    # cacheable against the model that gave it.
    model: str = ""


@dataclass
class ModelResponse:
    key: str
    data: dict | None = None
    error: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0
    attempts: int = 1
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.data is not None and not self.error


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Claude CLI
# ---------------------------------------------------------------------------


class ClaudeCliClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        timeout_s: int = 300,
        binary: str = "claude",
    ):
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.binary = binary

    def _argv(self, request: ModelRequest) -> list[str]:
        argv = [
            self.binary,
            "-p", request.prompt,
            "--model", request.model or self.model,
            "--effort", self.effort,
            "--output-format", "json",
            "--json-schema", json.dumps(request.schema, separators=(",", ":")),
            # No tools: the model reads what it was given and nothing else.
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if request.system:
            argv += ["--append-system-prompt", request.system]
        return argv

    def complete(self, request: ModelRequest) -> ModelResponse:
        start = time.time()
        try:
            proc = subprocess.run(
                self._argv(request),
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ModelResponse(
                key=request.key,
                error=f"timeout after {self.timeout_s}s",
                duration_s=time.time() - start,
                metadata=dict(request.metadata),
            )
        except Exception as exc:
            return ModelResponse(
                key=request.key,
                error=f"launch failed: {exc}",
                duration_s=time.time() - start,
                metadata=dict(request.metadata),
            )

        duration = time.time() - start
        return parse_cli_output(
            request, proc.stdout or "", proc.returncode, duration, proc.stderr or ""
        )


def parse_cli_output(
    request: ModelRequest, stdout: str, returncode: int, duration: float, stderr: str = ""
) -> ModelResponse:
    """Pull the structured object out of a CLI result envelope."""
    resp = ModelResponse(
        key=request.key, duration_s=duration, metadata=dict(request.metadata)
    )
    try:
        envelope = json.loads(stdout)
    except Exception:
        resp.error = f"exit {returncode}: output was not JSON"
        if stderr.strip():
            resp.error += f" ({stderr.strip()[:200]})"
        return resp

    resp.cost_usd = float(envelope.get("total_cost_usd") or 0.0)
    if envelope.get("is_error"):
        resp.error = str(envelope.get("result") or "model reported an error")[:300]
        return resp

    data = envelope.get("structured_output")
    if data is None:
        # Some configurations return the object as text in `result` instead.
        raw = envelope.get("result")
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = None
    if not isinstance(data, dict):
        resp.error = f"exit {returncode}: no structured output in response"
        return resp

    resp.data = data
    return resp


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


def _is_auth_error(error: str | None) -> bool:
    text = (error or "").lower()
    return "authentication_error" in text or "oauth access token" in text or "401" in text


class RetryingClient:
    """Retries transient failures. A schema-valid answer is never retried."""

    def __init__(self, inner: ModelClient, attempts: int = 3, backoff_s: float = 1.0):
        self.inner = inner
        self.attempts = attempts
        self.backoff_s = backoff_s

    def complete(self, request: ModelRequest) -> ModelResponse:
        last = ModelResponse(key=request.key, error="no attempt made")
        for attempt in range(1, self.attempts + 1):
            last = self.inner.complete(request)
            last.attempts = attempt
            if last.ok:
                return last
            # Auth failures are not transient. Retrying a 401 3x was what
            # stretched the hollow Aspirational run to ~90 minutes per task
            # with $0 spend.
            if _is_auth_error(last.error):
                return last
            if attempt < self.attempts:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))
        return last


class CachedClient:
    """Reads and writes through a `ResponseCache` using one shared key function."""

    def __init__(
        self,
        inner: ModelClient,
        cache: ResponseCache,
        *,
        model: str,
        effort: str,
        policy_version: str,
        refresh: bool = False,
    ):
        self.inner = inner
        self.cache = cache
        self.model = model
        self.effort = effort
        self.policy_version = policy_version
        self.refresh = refresh

    def _model_for(self, request: ModelRequest) -> str:
        return request.model or self.model

    def _key(self, request: ModelRequest) -> str:
        return compute_cache_key(
            prompt=request.prompt,
            schema=request.schema,
            model=self._model_for(request),
            effort=self.effort,
            policy_version=self.policy_version,
            system=request.system,
            sample_index=request.sample_index,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        if not self.cache.enabled:
            return self.inner.complete(request)
        cache_key = self._key(request)
        if not self.refresh:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return ModelResponse(
                    key=request.key,
                    data=hit.payload,
                    cost_usd=0.0,
                    cached=True,
                    metadata=dict(request.metadata),
                )
        resp = self.inner.complete(request)
        # Only successful responses are cached; caching an error would make a
        # transient failure permanent.
        if resp.ok:
            self.cache.put(
                cache_key,
                resp.data or {},
                request_key=request.key,
                model=self._model_for(request),
                effort=self.effort,
                prompt=request.prompt,
                schema=request.schema,
                policy_version=self.policy_version,
                cost_usd=resp.cost_usd,
            )
        return resp


class RoutedClient:
    """Sends named request kinds to a model other than the client's default.

    The audit's cost is not spread evenly over its checks and neither is its
    accuracy, so the cheap judge that is good enough for most of them can be
    wrong for one. This exists so that is a per-check decision rather than a
    whole-run one: a config can buy the strong model back for the check that
    needs it without paying for it everywhere.

    Keyed on the kind segment of the request key -- `task::kind::...` -- which is
    the same name the response cache stores and `CALL_NAME_TO_CHECK` maps to a
    check. Callers pass kinds rather than check ids to keep this file free of the
    stage vocabulary; the CLI owns that translation.

    Must sit outside `CachedClient`, so a routed request keys and stores under
    the model that answered it. Inside, it would read the default model's cached
    answer and report it as the routed model's.
    """

    def __init__(self, inner: ModelClient, model_by_kind: dict[str, str]):
        self.inner = inner
        self.model_by_kind = {k: v for k, v in model_by_kind.items() if v}

    @staticmethod
    def kind_of(key: str) -> str:
        parts = key.split("::")
        return parts[1] if len(parts) > 1 else ""

    def complete(self, request: ModelRequest) -> ModelResponse:
        # An explicit per-request model wins: routing is a default, not an
        # override of something a caller already decided.
        if request.model:
            return self.inner.complete(request)
        model = self.model_by_kind.get(self.kind_of(request.key))
        if not model:
            return self.inner.complete(request)
        return self.inner.complete(replace(request, model=model))


class ThrottledClient:
    """Caps concurrent calls across every stage and every task at once.

    Without this, concurrency is whatever the innermost pool happens to be, and
    the two failure modes pull in opposite directions. Stages that batch
    aggressively build few requests per task -- the informed stage builds 13 --
    so a per-task pool leaves most workers idle no matter how high `--workers`
    goes. Auditing several tasks at once fixes that, but then the real number of
    in-flight calls is tasks times workers, which overshoots the rate limit.

    Making the limit global decouples the two: how many tasks run at once becomes
    a scheduling choice, while this stays the one number that decides load.
    """

    def __init__(self, inner: ModelClient, max_in_flight: int):
        self.inner = inner
        self._sem = threading.Semaphore(max(1, max_in_flight))

    def complete(self, request: ModelRequest) -> ModelResponse:
        with self._sem:
            return self.inner.complete(request)


class FakeModelClient:
    """Deterministic stand-in for tests.

    `responder` maps a request to either a payload dict or a ModelResponse.
    """

    def __init__(
        self,
        responder: Callable[[ModelRequest], dict | ModelResponse | None] | dict | None = None,
    ):
        self.responder = responder
        self.calls: list[ModelRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.calls.append(request)
        result: Any = None
        if callable(self.responder):
            result = self.responder(request)
        elif isinstance(self.responder, dict):
            result = self.responder.get(request.key)
        if isinstance(result, ModelResponse):
            return result
        if result is None:
            return ModelResponse(
                key=request.key,
                error="fake client has no response for this key",
                metadata=dict(request.metadata),
            )
        return ModelResponse(
            key=request.key, data=result, metadata=dict(request.metadata)
        )

    @property
    def prompts(self) -> list[str]:
        return [c.prompt for c in self.calls]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def run_requests(
    client: ModelClient,
    requests: Iterable[ModelRequest],
    workers: int = 4,
    on_result: Callable[[ModelResponse], None] | None = None,
) -> list[ModelResponse]:
    """Run requests concurrently, returning results in submission order.

    Each request is independent by construction. Judgments are never batched into
    a single call, because sharing context lets one criterion's verdict influence
    the next.
    """
    reqs = list(requests)
    if not reqs:
        return []
    if workers <= 1:
        out = []
        for r in reqs:
            resp = client.complete(r)
            if on_result:
                on_result(resp)
            out.append(resp)
        return out

    results: list[ModelResponse | None] = [None] * len(reqs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.complete, r): i for i, r in enumerate(reqs)}
        # `as_completed`, not `futures.items()`: iterating in submission order
        # makes every already-finished result wait behind whichever request
        # happens to be slowest (a stuck retry can hold a 300s*3-attempt worst
        # case), so a batch that is 95% done still reports nothing until the
        # slow one resolves. Completion order surfaces each result -- and its
        # cache write, which happens inside `client.complete` -- the moment
        # it's actually ready.
        for future in as_completed(futures):
            i = futures[future]
            try:
                resp = future.result()
            except Exception as exc:
                resp = ModelResponse(key=reqs[i].key, error=f"worker crashed: {exc}")
            results[i] = resp
            if on_result:
                on_result(resp)
    return [r for r in results if r is not None]


def build_client(
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    timeout_s: int = 300,
    cache_path: str | None = None,
    policy_version: str = "",
    attempts: int = 3,
    refresh: bool = False,
    max_in_flight: int | None = None,
    model_by_kind: dict[str, str] | None = None,
) -> tuple[ModelClient, ResponseCache]:
    cache = ResponseCache(cache_path)
    client: ModelClient = RetryingClient(
        ClaudeCliClient(model=model, effort=effort, timeout_s=timeout_s), attempts=attempts
    )
    if cache.enabled:
        client = CachedClient(
            client,
            cache,
            model=model,
            effort=effort,
            policy_version=policy_version,
            refresh=refresh,
        )
    # Outside the cache, so a routed call keys under the model that answered it.
    if model_by_kind:
        client = RoutedClient(client, model_by_kind)
    # Outermost, so a cache hit costs nothing against the limit and a retry does
    # not have to queue behind unrelated work to finish.
    if max_in_flight:
        client = ThrottledClient(client, max_in_flight)
    return client, cache
