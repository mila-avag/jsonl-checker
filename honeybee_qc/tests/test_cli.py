"""End-to-end deterministic run over a JSONL batch."""

from __future__ import annotations

import dataclasses
import json

from honeybee_qc.cli import main, run_deterministic
from honeybee_qc.registry import REGISTRY
from honeybee_qc.rollup import roll_up_task
from honeybee_qc.tests.fixtures import make_task


def encode(task) -> dict:
    d = dataclasses.asdict(task)
    for key in ("model_a", "model_b"):
        if d[key]:
            d[key]["provider"] = d[key].pop("declared_provider")
    return d


def write_batch(tmp_path, tasks):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join(json.dumps(encode(t)) for t in tasks) + "\n", encoding="utf-8")
    return p


def test_every_dimension_appears_in_the_report_even_when_unimplemented():
    verdicts = run_deterministic(make_task())
    assert {v.check_id for v in verdicts} == set(REGISTRY)


def test_unimplemented_checks_are_not_evaluated_rather_than_clean():
    verdicts = {v.check_id: v for v in run_deterministic(make_task())}
    assert verdicts[220].band == "not_evaluated"
    assert verdicts[220].score is None
    assert verdicts[90].band == "clean"


def test_a_clean_task_rolls_up_clean_with_only_deterministic_checks_run():
    task = make_task()
    r = roll_up_task(task.task_id, run_deterministic(task))
    assert r.verdict == "clean"
    assert r.fail_checks == []


def test_cli_writes_a_report_and_attributes_each_failure(tmp_path, capsys):
    good = make_task("good")

    turn1 = make_task("turn1", key_turn=1)

    bad_link = make_task("badlink")
    bad_link.model_b.final_link = "https://gemini.google.com/app/private"

    inverted = make_task("inverted", dimension_rating_a=5, dimension_rating_b=2, likert=7)

    spam = make_task("spam")
    spam.rubric = []
    spam.criterion_ratings = []

    src = write_batch(tmp_path, [good, turn1, bad_link, inverted, spam])
    out = tmp_path / "report.json"
    assert main([str(src), "--out", str(out)]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    by_id = {t["task_id"]: t for t in report["tasks"]}

    assert by_id["good"]["verdict"] == "clean"
    assert by_id["turn1"]["fail_checks"] == [100]
    assert by_id["badlink"]["fail_checks"] == [90]
    assert by_id["inverted"]["fail_checks"] == [460]
    # An empty rubric alone (spam has no other hard failure here) must not
    # make the task unauditable: 90/96/100/460 all still pass for it, so it
    # rolls up clean, and check 1000 is not in its fail list.
    assert by_id["spam"]["verdict"] == "clean"
    assert 1000 not in by_id["spam"]["fail_checks"]

    counts = report["batch"]["task_counts"]
    assert counts["total"] == 5
    assert counts["auditable"] == 5
    # No task has verdict "unauditable" any more, so the bucket key is simply
    # absent from the counts rather than present at zero.
    assert counts.get("unauditable", 0) == 0


def test_a_rubric_empty_tasks_deterministic_checks_still_count(tmp_path):
    """90 never reads the rubric, so it counts for "spam" same as any task.

    No model stage ran in this deterministic-only call, so a check that only
    a stage can produce (e.g. 230) is not_evaluated -- and excluded from its
    own denominator -- for every task, unauditable or not.
    """
    good = make_task("good")
    spam = make_task("spam")
    spam.rubric = []
    spam.criterion_ratings = []

    src = write_batch(tmp_path, [good, spam])
    out = tmp_path / "report.json"
    main([str(src), "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    rate_90 = next(r for r in report["batch"]["check_rates"] if r["check_id"] == 90)
    assert rate_90["denominator"] == 2

    rate_230 = next(r for r in report["batch"]["check_rates"] if r["check_id"] == 230)
    assert rate_230["denominator"] == 0


def test_dry_run_reports_the_call_count_without_touching_a_model(tmp_path, capsys):
    src = write_batch(tmp_path, [make_task("a", n_criteria=20), make_task("b", n_criteria=10)])
    assert main([str(src), "--rubric-stage", "--rating-stage", "--dry-run"]) == 0

    err = capsys.readouterr().err
    # Rubric: (20+1) + (10+1).
    #
    # Rating is the blind pass plus an upper bound on the informed confirming pass
    # that checks 300 and 400 now run over whatever the blind pass disagreed about.
    # Blind: (40+12+1) + (20+12+1) = 86. Confirmation, worst case, is one call per
    # dimension plus one for the ranking: (12+1) x 2 = 26. A task whose two sides
    # agree everywhere spends none of it, so the dry run over-states rather than
    # surprising the operator with a bill.
    assert "rubric_calls=32" in err
    assert "rating_calls=112" in err
    assert "total=144" in err


def test_both_model_stages_run_and_land_in_one_report(tmp_path):
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient
    from honeybee_qc.tests.test_rating_stage import _responder as make_rating_client
    from honeybee_qc.tests.test_rubric_stage import stage_responder

    task = make_task("both", n_criteria=5, exchanges=6, key_turn=3)
    rating_client = make_rating_client()
    rubric_respond = stage_responder(task)

    def respond(req):
        if req.metadata.get("check_id") in (270, 300, 400):
            return rating_client.complete(req)
        return rubric_respond(req)

    payload = audit_batch(
        [task],
        client=FakeModelClient(respond),
        rubric_stage=True,
        rating_stage=True,
    )

    checks = {c["check_id"]: c for c in payload["tasks"][0]["checks"]}
    for check_id in (200, 230, 240, 250, 260, 270, 300, 400):
        assert checks[check_id]["band"] != "not_evaluated", check_id
    assert len(payload["rating_stage"]) == 1
    assert payload["rating_stage"][0]["calls"] == 10 + 12 + 1


def test_an_empty_rubric_alone_does_not_fail_check_1000():
    """`rubric_empty_only` must not read as a check-1000 fail.

    An empty rubric is expected for some batches and is explicitly carved out
    from the escape hatch (see `RUBRIC_EMPTY_REASON`'s docstring in
    `preflight.py`): it zeroes the rubric-dependent checks, but the task
    itself -- its conversation, its SxS pick -- is still fully auditable.
    """
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient

    spam = make_task("spam")
    spam.rubric = []
    spam.criterion_ratings = []

    payload = audit_batch([spam], client=FakeModelClient(lambda req: None))

    task = payload["tasks"][0]
    assert task["verdict"] != "unauditable"
    assert 1000 not in task["fail_checks"]
    checks = {c["check_id"]: c for c in task["checks"]}
    assert checks[1000]["band"] == "not_evaluated"
    assert "rubric is empty by design" in checks[1000]["measurement"]["notes"]
    assert "otherwise auditable" in checks[1000]["measurement"]["notes"]


def test_a_genuinely_unauditable_task_still_fails_check_1000():
    """A real integrity problem -- here, a missing SxS likert -- is not the
    `rubric_empty_only` carve-out and must still fail 1000 exactly as before,
    with the task's overall verdict staying "unauditable"."""
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient

    broken = make_task("broken")
    broken.sxs.likert = None

    payload = audit_batch([broken], client=FakeModelClient(lambda req: None))

    task = payload["tasks"][0]
    assert task["verdict"] == "unauditable"
    assert task["fail_checks"] == [1000]
    checks = {c["check_id"]: c for c in task["checks"]}
    assert checks[1000]["band"] == "fail"
    assert checks[1000]["error_code"] == "This task is unauditable"


def test_an_empty_rubric_plus_a_real_integrity_problem_still_fails_1000():
    """`rubric_empty_only` is exactly `reasons == [RUBRIC_EMPTY_REASON]`: add
    any other hard failure -- here, a duplicated task_id in the batch -- and
    the task is `fully_unauditable`, empty rubric or not."""
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient

    dup_a = make_task("dup")
    dup_a.rubric = []
    dup_a.criterion_ratings = []
    dup_b = make_task("dup")
    dup_b.rubric = []
    dup_b.criterion_ratings = []

    payload = audit_batch([dup_a, dup_b], client=FakeModelClient(lambda req: None))

    for task in payload["tasks"]:
        assert task["verdict"] == "unauditable"
        assert 1000 in task["fail_checks"]


def test_a_structurally_unauditable_task_never_spends_a_rating_call(tmp_path):
    """A non-rubric hard failure (here, a missing SxS likert) still blocks
    every model stage: the rubric is intact, but the submission itself is
    incomplete, so nothing downstream can be trusted either.
    """
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient

    broken = make_task("broken")
    broken.sxs.likert = None

    client = FakeModelClient(lambda req: None)
    payload = audit_batch([broken], client=client, rubric_stage=True, rating_stage=True)

    assert client.calls == []
    assert payload["tasks"][0]["verdict"] == "unauditable"
    checks = {c["check_id"]: c for c in payload["tasks"][0]["checks"]}
    assert checks[270]["band"] == "not_evaluated"
    assert checks[400]["band"] == "not_evaluated"
    assert "rating stage skipped" in checks[270]["measurement"]["notes"]


def test_a_rubric_empty_task_still_spends_rating_and_informed_calls(tmp_path):
    """Only the rubric-dependent checks (270/280) are withheld."""
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient
    from honeybee_qc.tests.test_rating_stage import _responder

    spam = make_task("spam")
    spam.rubric = []
    spam.criterion_ratings = []

    rating_fn = _responder().responder

    def respond(req):
        if req.metadata.get("check_id") in (270, 300, 400):
            return rating_fn(req)
        return {"confidence": "high"}  # generic informed-stage answer

    client = FakeModelClient(respond)
    payload = audit_batch(
        [spam],
        client=client,
        rubric_stage=True,
        rating_stage=True,
        informed_stage=True,
    )

    assert client.calls, "300/400 and the informed stage should still spend calls"
    assert all(r.metadata.get("check_id") != 270 for r in client.calls)
    # An empty rubric alone must not roll the task up as unauditable, and
    # must not put 1000 in its fail list -- see
    # `test_an_empty_rubric_alone_does_not_fail_check_1000` for the isolated
    # case; this test only re-confirms it holds once the rating and informed
    # stages are actually live.
    assert payload["tasks"][0]["verdict"] != "unauditable"
    checks = {c["check_id"]: c for c in payload["tasks"][0]["checks"]}
    assert checks[1000]["band"] == "not_evaluated"
    assert 1000 not in payload["tasks"][0]["fail_checks"]
    # Rubric-dependent: still not_evaluated, because there is no rubric.
    assert checks[270]["band"] == "not_evaluated"
    assert checks[220]["band"] == "not_evaluated"
    # Never read the rubric: really judged, not skipped by the gate.
    assert checks[300]["band"] != "not_evaluated"
    assert checks[400]["band"] != "not_evaluated"
    assert checks[110]["band"] != "not_evaluated"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_the_call_limit_holds_across_tasks_running_at_once():
    """--workers is a global cap, not a per-task one.

    Auditing several tasks concurrently is what keeps a large pool busy when a
    stage batches per model, but it would multiply in-flight calls by the number
    of tasks if the limit lived in each pool.
    """
    import threading

    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient, ThrottledClient
    from honeybee_qc.tests.test_rubric_stage import stage_responder

    in_flight = 0
    peak = 0
    lock = threading.Lock()

    tasks = [make_task(f"t{i}", n_criteria=5, exchanges=6, key_turn=3) for i in range(6)]
    responders = {t.task_id: stage_responder(t) for t in tasks}

    def respond(req):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            return responders[req.key.split("::")[0]](req)
        finally:
            with lock:
                in_flight -= 1

    audit_batch(
        tasks,
        client=ThrottledClient(FakeModelClient(respond), max_in_flight=4),
        rubric_stage=True,
        workers=8,
        task_workers=6,
    )
    assert peak <= 4


def test_running_tasks_in_parallel_does_not_reorder_the_report():
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient
    from honeybee_qc.tests.test_rubric_stage import stage_responder

    tasks = [make_task(f"t{i}", n_criteria=3, exchanges=6, key_turn=3) for i in range(5)]
    responders = {t.task_id: stage_responder(t) for t in tasks}

    def respond(req):
        return responders[req.key.split("::")[0]](req)

    def run(task_workers):
        return audit_batch(
            list(tasks),
            client=FakeModelClient(respond),
            rubric_stage=True,
            workers=4,
            task_workers=task_workers,
        )

    serial, parallel = run(1), run(5)
    assert [t["task_id"] for t in parallel["tasks"]] == [f"t{i}" for i in range(5)]
    assert [t["verdict"] for t in parallel["tasks"]] == [
        t["verdict"] for t in serial["tasks"]
    ]
    assert [s["task_id"] for s in parallel["rubric_stage"]] == [
        s["task_id"] for s in serial["rubric_stage"]
    ]


def test_one_task_raising_does_not_lose_the_rest_of_the_batch(monkeypatch):
    """A run costs hours, so a bug in one task must not discard sixteen others.

    The failure modelled here is a stage that raises before any call is made,
    like the independence guard refusing a prompt: a per-request error already
    comes back as a failed response instead of an exception.
    """
    from honeybee_qc import cli
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient
    from honeybee_qc.tests.test_rubric_stage import stage_responder

    tasks = [make_task(f"t{i}", n_criteria=3, exchanges=6, key_turn=3) for i in range(4)]
    responders = {t.task_id: stage_responder(t) for t in tasks}

    def respond(req):
        return responders[req.key.split("::")[0]](req)

    real_stage = cli.run_rubric_stage

    def flaky_stage(task, client, policy, **kw):
        if task.task_id == "t2":
            raise RuntimeError("boom")
        return real_stage(task, client, policy, **kw)

    monkeypatch.setattr(cli, "run_rubric_stage", flaky_stage)

    payload = audit_batch(
        list(tasks),
        client=FakeModelClient(respond),
        rubric_stage=True,
        workers=4,
        task_workers=4,
    )

    assert [t["task_id"] for t in payload["tasks"]] == [f"t{i}" for i in range(4)]
    assert [e["task_id"] for e in payload["stage_errors"]] == ["t2"]
    assert payload["stage_errors"][0]["stage"] == "rubric"
    # The task survives with its rubric checks unscored rather than absent.
    checks = {
        c["check_id"]: c for c in next(t for t in payload["tasks"] if t["task_id"] == "t2")["checks"]
    }
    assert checks[200]["band"] == "not_evaluated"
    assert "rubric stage failed" in payload["stage_errors"][0]["error"] or "boom" in (
        payload["stage_errors"][0]["error"]
    )
    others = [t for t in payload["tasks"] if t["task_id"] != "t2"]
    assert all(t["verdict"] != "unauditable" for t in others)


def test_check_210_is_reported_exactly_once_not_twice():
    """run_deterministic used to hardcode a 210 verdict of its own on top of the
    one run_rubric_stage appends, doubling 210's denominator in every report."""
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient
    from honeybee_qc.tests.test_rubric_stage import stage_responder

    task = make_task("one_ten", n_criteria=5, exchanges=6, key_turn=3)
    payload = audit_batch(
        [task], client=FakeModelClient(stage_responder(task)), rubric_stage=True
    )

    checks = [c for c in payload["tasks"][0]["checks"] if c["check_id"] == 210]
    assert len(checks) == 1
    assert checks[0]["band"] == "clean"


def test_check_210_is_not_evaluated_without_the_rubric_stage():
    """No rubric stage means nobody ever looked at the L2 labels; 210 must not
    default to a speculative clean verdict."""
    verdicts = {v.check_id: v for v in run_deterministic(make_task())}
    assert verdicts[210].band == "not_evaluated"


def test_a_batch_with_no_failures_reports_no_stage_errors():
    from honeybee_qc.cli import audit_batch
    from honeybee_qc.llm import FakeModelClient
    from honeybee_qc.tests.test_rubric_stage import stage_responder

    task = make_task("ok", n_criteria=3, exchanges=6, key_turn=3)
    payload = audit_batch(
        [task], client=FakeModelClient(stage_responder(task)), rubric_stage=True
    )
    assert payload["stage_errors"] == []


def test_report_carries_preflight_warnings_for_an_otherwise_clean_task(tmp_path):
    task = make_task("warned", exchanges=6, key_turn=5)
    src = write_batch(tmp_path, [task])
    out = tmp_path / "report.json"
    main([str(src), "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    pre = report["preflight"][0]
    assert pre["ok"]
    assert any("fewer than the 7 turns" in w for w in pre["warnings"])


# ---------------------------------------------------------------------------
# Sampling configuration on the command line
# ---------------------------------------------------------------------------


def test_dry_run_prices_the_sample_configuration_it_was_asked_for(tmp_path, capsys):
    """A dry run has to show the sampled call count, not the judgment count. The
    sample count is the one knob that multiplies the bill, so a dry run that
    reported judgments would understate a three-draw run threefold."""
    task = make_task("priced", exchanges=6, key_turn=3)
    task.criterion_ratings[0].score = 0
    src = write_batch(tmp_path, [task])

    main([str(src), "--informed-stage", "--dry-run"])
    sampled = capsys.readouterr().err

    main([str(src), "--informed-stage", "--dry-run", "--no-sampling"])
    single = capsys.readouterr().err

    def calls(text: str) -> int:
        line = next(l for l in text.splitlines() if "informed_calls=" in l)
        return int(line.split("informed_calls=")[1].split()[0])

    assert calls(sampled) > calls(single)
    assert "for repeats" in sampled
    assert "+0 for repeats" in single
    # Without a cache to price against, the dry run says so rather than printing a
    # dollar figure it cannot support.
    assert "no per-call prices available" in sampled
