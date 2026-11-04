"""Static receipt safety and accounting, using synthetic ledgers and no model or network calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import pytest

from receipts.contracts import (
    CatchKind,
    FailureType,
    Finding,
    Lens,
    Outcome,
    ReplayKey,
    Repro,
    RunResult,
    Side,
    StageCost,
    Status,
    UncheckedReason,
)
from receipts.surface.ledger import JsonlLedger, new_entry
from receipts.surface.receipt_command import register
from receipts.surface.receipt_html import CSP, ReceiptSummary, build_view, read_summary, render_receipt

AT = datetime(2026, 10, 30, 12, tzinfo=UTC)
CLAIM = "nothing_else_changed"
SPARK = "muse-spark-1.3"
WONDER = "muse-wonder-1.0"
SHA_A, SHA_B = "a" * 40, "b" * 40
CANARY = (
    '<script>alert(1)</script> "><img src=x onerror=alert(1)> '
    "![x](https://evil.example/a.png) [c](http://evil.example) javascript:alert(1) "
    "</code></pre><a href=//evil.example> \u202e\u200b"
)
# The snapshot is embedded because this lane owns exactly this test file, with no extra fixture files.
GOLDEN_SHA256 = "ecaba8959a42ad0ed9a964bdf00343fd6ead4c0693089571a4fa67031e644c4d"


class Page(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.head_children: list[str] = []
        self.text: list[str] = []
        self.code: list[str] = []
        self.rows: list[list[str]] = []
        self.css = ""
        self._head = False
        self._style = False
        self._code = False
        self._cell = False
        self.feed(source)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if self._head:
            self.head_children.append(tag)
        if tag == "head":
            self._head = True
        if tag == "style":
            self._style = True
        if tag == "code":
            self._code = True
            self.code.append("")
        if tag == "tr":
            self.rows.append([])
        if tag in ("td", "th"):
            self._cell = True
            self.rows[-1].append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._head = False
        if tag == "style":
            self._style = False
        if tag == "code":
            self._code = False
        if tag in ("td", "th"):
            self._cell = False

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self._style:
            self.css += data
        if self._code:
            self.code[-1] += data
        if self._cell:
            self.rows[-1][-1] += data

    @property
    def content(self) -> str:
        return " ".join(self.text)


def cost(stage: str, model: str | None, micros: int, seconds: float = 1) -> StageCost:
    return StageCost(
        stage=stage,
        model=model,
        input_tokens=100,
        cached_tokens=25,
        output_tokens=50,
        reasoning_tokens=10,
        micro_usd=micros,
        seconds=seconds,
    )


def finding(
    status: Status = Status.BLOCKER,
    *,
    target: str = "t1",
    name: str = "a1",
    text: str = "The empty list keeps working.",
    kind: CatchKind = CatchKind.REGRESSION,
    costs: tuple[StageCost, ...] = (),
    wall: float | None = 12,
) -> Finding:
    fail = RunResult(
        side=Side.PATCH,
        outcome=Outcome.FAIL,
        failure_type=FailureType.ASSERTION,
        fresh_container=True,
        duration_s=1,
        seed=7,
        stdout_tail=text,
        stderr_tail=text,
        problem=text,
    )
    passed = RunResult(
        side=Side.BASE if kind is CatchKind.REGRESSION else Side.SURROGATE,
        outcome=Outcome.PASS,
        fresh_container=True,
        duration_s=1,
        seed=7,
    )
    blocker = status is Status.BLOCKER
    return Finding(
        id=f"{target}/{CLAIM}/{name}",
        target_id=target,
        claim_id=CLAIM,
        lens=Lens.REGRESSION,
        status=status,
        kind=kind if blocker else None,
        unchecked_reason=UncheckedReason.TIMEOUT if status is Status.UNCHECKED else None,
        repro=Repro(
            id="r1",
            attack_id=name,
            claim_id=CLAIM,
            lens=Lens.REGRESSION,
            kind=kind,
            files={"tests/test_repro.py": text},
            command=("python", "-m", "pytest", text),
            cases=(text,),
            quote=text,
        )
        if blocker
        else None,
        runs=(fail,) * 5 + (passed,) * (5 if kind is CatchKind.REGRESSION else 1) if blocker else (),
        quote=text if blocker else None,
        detail=text,
        refutation=text if blocker else None,
        replay=ReplayKey(base_sha=SHA_A, patch_sha=SHA_B, harness_hash="h1", seed=7, image_digest="sha256:fixture")
        if blocker
        else None,
        costs=costs,
        wall_seconds=wall,
    )


def changed(original: Finding, **updates: object) -> Finding:
    return Finding.model_validate({**original.model_dump(), **updates})


def write(path: Path, *findings: Finding) -> JsonlLedger:
    ledger = JsonlLedger(path)
    for offset, item in enumerate(findings):
        ledger.append(new_entry(item, at=AT + timedelta(seconds=offset)))
    return ledger


def summary_data(**updates: object) -> dict[str, object]:
    return {
        "target_id": "t1",
        "tier": "A",
        "claims": [
            {
                "id": CLAIM,
                "type": CLAIM,
                "text": "The empty list keeps working.",
                "span": {"source": "pr_body", "text": "The empty list keeps working."},
            }
        ],
        "plan": {"unchecked": [], "attacks": [], "target_id": "t1"},
        "costs": [
            cost("intake", SPARK, 12_345, 2).model_dump(),
            cost("plan", WONDER, 40_000, 3).model_dump(),
        ],
        "planner_model": WONDER,
        "planner_fallback": True,
        "refuter_model": SPARK,
        "wall_seconds": 15,
        "ignored_future_field": {"not": "part of this lane"},
        **updates,
    }


def save_summary(path: Path, **updates: object) -> Path:
    path.write_text(json.dumps(summary_data(**updates)), encoding="utf-8")
    return path


def call(*argv: str) -> int:
    parser = argparse.ArgumentParser(prog="receipts")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(list(argv))
    result: int = args.func(args)
    return result


def test_r16_canary_is_text_only(tmp_path: Path) -> None:
    original = finding(text=CANARY, costs=(cost(CANARY, CANARY, 123_456),))
    assert original.repro is not None and original.replay is not None
    repro = original.repro.model_copy(update={"files": {f"tests/{CANARY}.py": CANARY}})
    replay = original.replay.model_copy(update={"harness_hash": CANARY, "image_digest": CANARY})
    ledger = write(tmp_path / "arm.jsonl", changed(original, repro=repro, replay=replay))
    supplied = ReceiptSummary.model_validate(
        summary_data(
            planner_model=CANARY,
            refuter_model=CANARY,
            plan={"unchecked": [{"what": CANARY, "reason": "budget", "detail": CANARY}]},
        )
    )
    source = render_receipt(build_view("t1", ledger, summary=supplied))
    page = Page(source)
    allowed = {
        "html",
        "head",
        "meta",
        "title",
        "style",
        "body",
        "main",
        "header",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "dl",
        "dt",
        "dd",
        "div",
        "section",
        "table",
        "caption",
        "thead",
        "tr",
        "th",
        "tbody",
        "td",
        "pre",
        "code",
        "article",
        "ul",
        "li",
    }
    assert {tag for tag, _ in page.tags} <= allowed
    assert not {tag for tag, _ in page.tags} & {"script", "img", "a", "link", "iframe", "object", "form", "base"}
    for _, attrs in page.tags:
        assert not {"href", "src", "action", "formaction"} & attrs.keys()
        assert not any(name.startswith("on") for name in attrs)
        assert not any("alert(1)" in (value or "") for value in attrs.values())
    assert not re.search(r"http|www\.|javascript:", page.content, re.IGNORECASE)
    assert "\\u202e\\u200b" in page.content
    assert "\u202e" not in source and "\u200b" not in source
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in source
    assert "&quot;&gt;&lt;img" in source
    assert any("<script>alert(1)</script>" in block for block in page.code)
    assert "receipts.falden.ai" not in source


def test_csp_is_first_head_child_and_css_cannot_fetch(tmp_path: Path) -> None:
    page = Page(render_receipt(build_view("t1", write(tmp_path / "arm.jsonl", finding()))))
    assert page.head_children[0] == "meta"
    first_meta = next(attrs for tag, attrs in page.tags if tag == "meta")
    assert first_meta == {"http-equiv": "Content-Security-Policy", "content": CSP}
    assert CSP == "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; form-action 'none'"
    assert len([tag for tag, _ in page.tags if tag == "style"]) == 1
    assert "url(" not in page.css.lower() and "@import" not in page.css.lower()
    assert "prefers-color-scheme: dark" in page.css
    assert "overflow: auto" in page.css


def test_header_keeps_full_shas_version_and_latest_time(tmp_path: Path) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(), finding(name="a2"))
    page = Page(render_receipt(build_view("t1", ledger, summary=ReceiptSummary.model_validate(summary_data()))))
    for value in ("Falden Receipts", "t1", SHA_A, SHA_B, "2026-10-30T12:00:01.000000Z", "Tier", "A"):
        assert value in page.content
    assert ledger.entries()[0].receipts_version in page.code


def test_every_stage_model_tokens_and_exact_dollars(tmp_path: Path) -> None:
    ledger = write(
        tmp_path / "arm.jsonl",
        finding(costs=(cost("attack", SPARK, 1_234_567, 4), cost("execute", None, 1, 2))),
        finding(name="a2", costs=(cost("attack", WONDER, 9_000, 1), cost("attack", SPARK, 33, 2))),
    )
    view = build_view("t1", ledger)
    page = Page(render_receipt(view))
    for stage in view.rollup.stages:
        expected = [
            stage.stage,
            stage.model or "code",
            str(stage.input_tokens),
            str(stage.cached_tokens),
            str(stage.output_tokens),
            str(stage.reasoning_tokens),
        ]
        row = next(row for row in page.rows if row[:2] == expected[:2])
        assert row[:6] == expected
    assert ["attack", SPARK, "200", "50", "100", "20", "$1.234600", "6.000"] in page.rows
    assert ["execute", "code", "100", "25", "50", "10", "$0.000001", "2.000"] in page.rows


def test_summary_costs_are_opt_in_and_added_once(tmp_path: Path) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(costs=(cost("attack", SPARK, 100_000),)))
    summary_path = save_summary(tmp_path / "summary.json")
    without = Page(render_receipt(build_view("t1", ledger)))
    assert "Intake and planning costs are not shown" in without.content
    assert "outside the ledgers" in without.content
    assert not any(row[0] in ("intake", "plan") for row in without.rows)
    assert "Target cost: $0.100000 (ledger entries only)" in without.content
    with_summary = Page(render_receipt(build_view("t1", ledger, summary=read_summary(summary_path))))
    assert ["intake", SPARK, "100", "25", "50", "10", "$0.012345", "2.000"] in with_summary.rows
    assert ["plan", WONDER, "100", "25", "50", "10", "$0.040000", "3.000"] in with_summary.rows
    assert "Target cost: $0.152345 (including summary costs)" in with_summary.content
    assert "Planner fell back: yes" in with_summary.content
    assert SPARK in with_summary.content and WONDER in with_summary.content
    assert "Summary wall time: 15.000 s" in with_summary.content
    assert "ignored_future_field" not in with_summary.content


def test_summary_and_ledger_same_stage_model_are_combined(tmp_path: Path) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(costs=(cost("plan", WONDER, 6),)))
    summary = ReceiptSummary.model_validate(summary_data())
    page = Page(render_receipt(build_view("t1", ledger, summary=summary)))
    plans = [row for row in page.rows if row[:2] == ["plan", WONDER]]
    assert plans == [["plan", WONDER, "200", "50", "100", "20", "$0.040006", "4.000"]]


def test_large_integer_money_never_rounds_through_float(tmp_path: Path) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(costs=(cost("attack", SPARK, 123_456_789_012_345_678),)))
    page = Page(render_receipt(build_view("t1", ledger)))
    assert "Target cost: $123,456,789,012.345678" in page.content


def test_not_checked_contains_every_finding_and_planned_lens(tmp_path: Path) -> None:
    ledger = write(
        tmp_path / "arm.jsonl",
        finding(Status.UNCHECKED, name="timeout", text="The runner timed out."),
        changed(
            finding(Status.UNCHECKED, name="budget", text="The cap was reached."),
            lens=Lens.CORRECTNESS,
            unchecked_reason=UncheckedReason.BUDGET,
        ),
    )
    summary = ReceiptSummary.model_validate(
        summary_data(
            plan={
                "unchecked": [
                    {"what": "differential", "reason": "runtime", "detail": "No comparable runtime."},
                    {"what": "mutation", "reason": "no_env", "detail": "No mutation environment."},
                ]
            }
        )
    )
    page = Page(render_receipt(build_view("t1", ledger, summary=summary)))
    section = page.content.split("Not checked", 1)[1].split("Cost and latency", 1)[0]
    for value in (
        "regression",
        "timeout",
        "The runner timed out.",
        "correctness",
        "budget",
        "The cap was reached.",
        "differential",
        "runtime",
        "No comparable runtime.",
        "mutation",
        "no_env",
        "No mutation environment.",
    ):
        assert value in section


def test_not_checked_empty_is_explicit(tmp_path: Path) -> None:
    page = Page(render_receipt(build_view("t1", write(tmp_path / "arm.jsonl", finding()))))
    assert "Not checked" in page.content and "Nothing was left unchecked." in page.content


def test_planned_detail_truncation_points_to_summary(tmp_path: Path) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(Status.UNCHECKED))
    summary = ReceiptSummary.model_validate(
        summary_data(plan={"unchecked": [{"what": "mutation", "reason": "budget", "detail": "z" * 2_001}]})
    )
    page = Page(render_receipt(build_view("t1", ledger, summary=summary)))
    assert "1 more characters cut; the rest is in the summary" in page.content
    assert "receipts replay t1)" not in page.content


@pytest.mark.parametrize("use_run", [False, True])
def test_one_target_percentiles_need_more_data(tmp_path: Path, use_run: bool) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(costs=(cost("attack", SPARK, 20_000),)))
    page = Page(render_receipt(build_view("t1", ledger, run_ledger=ledger if use_run else None)))
    assert "p50/p95 need more than one target (n=1)" in page.content
    assert "Target cost: $0.020000" in page.content
    assert "Target latency (wall): 12.000 s" in page.content


@pytest.mark.parametrize("basis", ["wall", "summed_stage_seconds", "mixed"])
def test_run_percentiles_are_nearest_rank_and_basis_is_labeled(tmp_path: Path, basis: str) -> None:
    ledger = write(
        tmp_path / "run.jsonl",
        finding(
            target="t1",
            costs=(cost("attack", SPARK, 30_000, 30),),
            wall=30 if basis != "summed_stage_seconds" else None,
        ),
        finding(target="t2", costs=(cost("attack", SPARK, 10_000, 10),), wall=10 if basis == "wall" else None),
        finding(target="t3", costs=(cost("attack", SPARK, 20_000, 20),), wall=20 if basis == "wall" else None),
    )
    page = Page(render_receipt(build_view("t1", ledger, run_ledger=ledger)))
    assert "Run cost: p50 $0.020000; p95 $0.030000 (n=3)" in page.content
    assert "p50 20.000 s; p95 30.000 s (n=3)" in page.content
    assert f"({basis})" in page.content
    if basis != "wall":
        assert "Run latency" not in page.content


def test_later_refutation_hides_blocker_but_keeps_all_costs(tmp_path: Path) -> None:
    first = finding(costs=(cost("attack", SPARK, 100),))
    last = finding(Status.REFUTED, costs=(cost("refute", WONDER, 200),), text="It was a fixture issue.")
    ledger = write(tmp_path / "arm.jsonl", first, last)
    view = build_view("t1", ledger)
    page = Page(render_receipt(view))
    assert len(view.entries) == 1 and view.entries[0].finding.status is Status.REFUTED
    assert "No blockers." in page.content
    assert "regression catch" not in page.content
    assert "blocker: 0" in page.content and "refuted: 1" in page.content
    assert "It was a fixture issue." in page.content
    assert "Target cost: $0.000300" in page.content
    assert SHA_A in page.content and SHA_B in page.content


@pytest.mark.parametrize(
    ("status", "verdict"),
    [
        (Status.BLOCKER, "1 blocker"),
        (Status.SUSPECTED, "suspected only"),
        (Status.REFUTED, "nothing caught"),
        (Status.UNCHECKED, "nothing checked"),
    ],
)
def test_strongest_verdict(tmp_path: Path, status: Status, verdict: str) -> None:
    page = render_receipt(build_view("t1", write(tmp_path / "arm.jsonl", finding(status))))
    assert f'<p class="verdict">{verdict}</p>' in page


def test_all_other_statuses_keep_their_lens_and_detail(tmp_path: Path) -> None:
    statuses = [s for s in Status if s not in (Status.BLOCKER, Status.UNCHECKED)]
    ledger = write(
        tmp_path / "arm.jsonl",
        *(finding(status, name=status.value, text=f"Reason for {status.value}.") for status in statuses),
    )
    page = Page(render_receipt(build_view("t1", ledger)))
    section = page.content.split("Suspected, flaky, refuted and the rest", 1)[1].split("Not checked", 1)[0]
    for status in statuses:
        assert status.value in section
        assert f"Reason for {status.value}." in section
    assert section.count("regression") == len(statuses)


@pytest.mark.parametrize("kind", list(CatchKind))
def test_blocker_kind_claim_runs_and_replay(tmp_path: Path, kind: CatchKind) -> None:
    item = finding(kind=kind)
    page = Page(render_receipt(build_view("t1", write(tmp_path / "arm.jsonl", item))))
    phrase = {
        CatchKind.REGRESSION: "regression catch",
        CatchKind.FIX_VALIDITY: "fix-validity catch",
        CatchKind.BEHAVIOR: "behavior catch",
    }[kind]
    for value in (phrase, item.id, item.quote, "Repro files", "Command", "Replay key", "h1", "sha256:fixture"):
        assert value is not None and value in page.content
    assert len([row for row in page.rows if row[:2] == ["patch", "fail"]]) == 5
    assert ["patch", "fail", "assertion", "yes", "7"] in page.rows
    assert "A different model's refutation, which did not hold" in page.content


def test_file_and_output_limits_after_hidden_char_expansion(tmp_path: Path) -> None:
    original = finding(text="x" * 3_001)
    assert original.repro is not None
    files = {f"test_{n}.py": "\u200b" * 2_000 for n in range(6)}
    item = changed(original, repro=original.repro.model_copy(update={"files": files}))
    page = Page(render_receipt(build_view("t1", write(tmp_path / "arm.jsonl", item))))
    file_blocks = [text for text in page.code if text.startswith("File:")]
    assert len(file_blocks) == 4
    assert all(len(text.split("\n", 1)[1].split("\n...", 1)[0]) <= 8_000 for text in file_blocks)
    assert sum(len(text.split("\n...", 1)[0]) for text in file_blocks) <= 30_000
    assert "2 files cut; receipts replay" in page.content
    assert "1 more characters cut; receipts replay" in page.content
    assert any("x" * 3_000 in text for text in page.code)
    assert not any("x" * 3_001 in text for text in page.code)


def test_same_inputs_match_embedded_golden(tmp_path: Path) -> None:
    ledger = write(tmp_path / "arm.jsonl", finding(Status.SUSPECTED, costs=(cost("attack", SPARK, 123_456),)))
    first = render_receipt(build_view("t1", ledger)).encode("utf-8")
    second = render_receipt(build_view("t1", ledger)).encode("utf-8")
    assert first == second
    assert hashlib.sha256(first).hexdigest() == GOLDEN_SHA256


def test_command_writes_self_contained_page(tmp_path: Path) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    summary = save_summary(tmp_path / "summary.json")
    output = tmp_path / "receipt.html"
    assert (
        call(
            "receipt",
            "html",
            "t1",
            "--ledger",
            str(path),
            "--summary",
            str(summary),
            "--run-ledger",
            str(path),
            "--out",
            str(output),
        )
        == 0
    )
    assert output.read_text().startswith("<!doctype html>\n")
    assert "Planner fell back: yes" in Page(output.read_text()).content


def test_unknown_target_exits_three_and_leaves_output_alone(tmp_path: Path) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    output = tmp_path / "receipt.html"
    output.write_text("keep this")
    assert call("receipt", "html", "missing", "--ledger", str(path), "--out", str(output)) == 3
    assert output.read_text() == "keep this"


def test_torn_tail_is_reported_once_and_never_changed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    with path.open("ab") as stream:
        stream.write(b'{"torn":')
    before = path.read_bytes()
    assert call("receipt", "html", "t1", "--ledger", str(path), "--out", str(tmp_path / "receipt.html")) == 0
    err = capsys.readouterr().err
    assert err.count("skipped torn last line") == 1
    assert "file is unchanged" in err
    assert path.read_bytes() == before
    assert not path.with_suffix(".jsonl.torn").exists()


@pytest.mark.parametrize("bad", ["missing", "corrupt", "summary", "mismatch", "output"])
def test_bad_input_exits_two_without_writing(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    output = tmp_path / "receipt.html"
    extra: list[str] = []
    if bad == "missing":
        path = tmp_path / "missing.jsonl"
    elif bad == "corrupt":
        path.write_text("not a ledger\n")
    elif bad in ("summary", "mismatch"):
        summary = save_summary(tmp_path / "summary.json", target_id="other" if bad == "mismatch" else "t1")
        if bad == "summary":
            summary.write_text('{"target_id": "t1", "wall_seconds": -1}')
        extra = ["--summary", str(summary)]
    elif bad == "output":
        output = tmp_path / "missing-directory" / "receipt.html"
    assert call("receipt", "html", "t1", "--ledger", str(path), *extra, "--out", str(output)) == 2
    assert not output.exists()


@pytest.mark.parametrize("seconds", [float("inf"), float("nan"), 1e308])
def test_invalid_or_overflowing_summary_seconds_exit_two(tmp_path: Path, seconds: float) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    output = tmp_path / "receipt.html"
    costs = [cost("intake", SPARK, 1).model_dump() for _ in range(2)]
    for item in costs:
        item["seconds"] = seconds
    summary = save_summary(tmp_path / "summary.json", costs=costs)
    assert call("receipt", "html", "t1", "--ledger", str(path), "--summary", str(summary), "--out", str(output)) == 2
    assert not output.exists()


def test_empty_run_ledger_is_bad_input(tmp_path: Path) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    run = tmp_path / "empty.jsonl"
    JsonlLedger(run)
    output = tmp_path / "receipt.html"
    assert call("receipt", "html", "t1", "--ledger", str(path), "--run-ledger", str(run), "--out", str(output)) == 2
    assert not output.exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_output_cannot_overwrite_ledger(tmp_path: Path, alias: str) -> None:
    path = tmp_path / "arm.jsonl"
    write(path, finding())
    output = path if alias == "same" else tmp_path / "alias.html"
    if alias == "symlink":
        output.symlink_to(path)
    elif alias == "hardlink":
        output.hardlink_to(path)
    before = path.read_bytes()
    assert call("receipt", "html", "t1", "--ledger", str(path), "--out", str(output)) == 2
    assert path.read_bytes() == before


def luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
    return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


@pytest.mark.parametrize(
    ("foreground", "background"),
    [("#191919", "#EDECEB"), ("#2200FF", "#EDECEB"), ("#EDECEB", "#191919"), ("#9B8FFF", "#191919")],
)
def test_theme_text_colors_meet_aa(foreground: str, background: str) -> None:
    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    assert (light + 0.05) / (dark + 0.05) >= 4.5
