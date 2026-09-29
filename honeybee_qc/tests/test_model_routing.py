"""Per-check model routing, and the cache isolation it depends on.

Routing exists because neither cost nor accuracy is spread evenly over the
checks: on the qc_val31 run a cheap judge matched human QC on the audit as a
whole while over-firing one check, and the fix for that should not be to pay the
strong model's price on all twenty-five.

The property worth defending is the cache one. A routed request that reads the
default model's stored answer would report one model's judgment under another's
name, which is worse than not routing at all, because the run would look like it
had bought the strong model and would not have.
"""

from __future__ import annotations

import dataclasses
import sqlite3

from honeybee_qc.cache import ResponseCache
from honeybee_qc.informed_stages import CALL_NAME_TO_CHECK
from honeybee_qc.llm import (
    CachedClient,
    FakeModelClient,
    ModelRequest,
    RoutedClient,
)


def _request(kind: str, task: str = "t1") -> ModelRequest:
    return ModelRequest(
        key=f"{task}::{kind}::A", prompt="p", schema={"type": "object"}
    )


def _routed(inner, **routes: str) -> RoutedClient:
    return RoutedClient(inner, dict(routes))


def test_only_the_named_kind_is_routed():
    inner = FakeModelClient(lambda r: {"ok": True})
    client = _routed(inner, dimension_turns="sonnet")

    client.complete(_request("dimension_turns"))
    client.complete(_request("dimension_justifications"))

    assert [c.model for c in inner.calls] == ["sonnet", ""]


def test_an_explicit_request_model_is_not_overridden():
    """Routing is a default. A caller that already chose keeps its choice."""
    inner = FakeModelClient(lambda r: {"ok": True})
    client = _routed(inner, dimension_turns="sonnet")

    client.complete(dataclasses.replace(_request("dimension_turns"), model="opus"))

    assert [c.model for c in inner.calls] == ["opus"]


def test_a_key_with_no_kind_segment_is_left_alone():
    inner = FakeModelClient(lambda r: {"ok": True})
    client = _routed(inner, dimension_turns="sonnet")

    client.complete(ModelRequest(key="bare", prompt="p", schema={}))

    assert [c.model for c in inner.calls] == [""]


def test_a_routed_call_does_not_read_the_default_models_cached_answer(tmp_path):
    """The load-bearing property, and the reason routing sits outside the cache.

    Same prompt, same schema, same check: the only difference is who is being
    asked, and that has to be enough to miss.
    """
    inner = FakeModelClient(lambda r: {"answered_by": r.model or "default"})
    cache = ResponseCache(tmp_path / "c.db")
    cached = CachedClient(inner, cache, model="haiku", effort="medium", policy_version="v1")

    request = _request("dimension_turns")

    # Unrouted first, so the default model's answer is the one already stored.
    plain = CachedClient(inner, cache, model="haiku", effort="medium", policy_version="v1")
    assert plain.complete(request).data == {"answered_by": "default"}

    routed = RoutedClient(cached, {"dimension_turns": "sonnet"})
    assert routed.complete(request).data == {"answered_by": "sonnet"}
    assert cache.hits == 0, "a routed call read the cheaper model's answer"
    assert len(inner.calls) == 2

    # And the routed answer is itself cached, under its own model.
    assert routed.complete(request).data == {"answered_by": "sonnet"}
    assert cache.hits == 1
    assert len(inner.calls) == 2

    # Read the file rather than the cache's own handle: what matters is that two
    # rows landed, attributed to the two models that actually answered.
    cache.close()
    with sqlite3.connect(tmp_path / "c.db") as con:
        stored = {row[0] for row in con.execute("select model from response_cache")}
    assert stored == {"haiku", "sonnet"}


def test_routing_a_check_covers_every_call_kind_it_owns():
    """450 asks two separate questions and both have to move together.

    A config that routed the dimension justifications and left the ranking one
    on the cheap model would report a single check answered by two judges, and
    the aggregation across them assumes one.
    """
    kinds = {k for k, c in CALL_NAME_TO_CHECK.items() if c == 450}
    assert kinds == {"dimension_justifications", "ranking_justification"}

    inner = FakeModelClient(lambda r: {"ok": True})
    client = RoutedClient(inner, {k: "sonnet" for k in kinds})
    for kind in sorted(kinds):
        client.complete(_request(kind))

    assert {c.model for c in inner.calls} == {"sonnet"}


def test_an_empty_route_target_is_ignored_rather_than_sent_as_blank():
    inner = FakeModelClient(lambda r: {"ok": True})
    client = RoutedClient(inner, {"dimension_turns": ""})

    client.complete(_request("dimension_turns"))

    assert [c.model for c in inner.calls] == [""]
