"""SK3's surrogate fixer over scripted fakes of the model client and a sandbox that reads its workspace back: no
model, no Docker and no network. Each test drives one rule: what the fixer is shown, where its diff comes from, which
diffs are refused, and what makes a surrogate plausible."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from receipts.agentloop.loop import ATTACKER_CAP_MICRO_USD, SHELL
from receipts.agentloop.surrogate import (
    DIFF_CAP_BYTES,
    DIFF_PATH,
    DIFF_SCRIPT,
    GIT_PREFLIGHT,
    SURROGATE_MODEL,
    SURROGATE_STAGE,
    TEMPERATURES,
    LoopSurrogateBuilder,
    WorkspaceReader,
    fixer_messages,
)
from receipts.attack.classify import SuiteRun
from receipts.contracts import (
    SURROGATE_CAP_MICRO_USD,
    CaseResult,
    Claim,
    ClaimType,
    FailureType,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    Outcome,
    RunnerKind,
    RunResult,
    Side,
    Span,
    SpanSource,
    StageCost,
    Surrogate,
    SurrogateBuilder,
    Target,
    Tier,
    UncheckedReason,
    Usage,
    Workspace,
)

SENTINELS = {
    "patch": "SENTINEL_PATCH_7f3a",
    "body": "SENTINEL_BODY_19c2",
    "summary": "SENTINEL_SUMMARY_5e81",
    "commits": "SENTINEL_COMMITS_aa04",
    "pr_test": "test_sentinel_pr_added_e2d9",
    "claim_text": "SENTINEL_CLAIM_TEXT_0b6d",
}
AGENT_PATCH = f"""diff --git a/src/widgets/render.py b/src/widgets/render.py
--- a/src/widgets/render.py
+++ b/src/widgets/render.py
@@ -1,2 +1,2 @@
 def render(items):
-    return "".join(items)
+    return ",".join(items)  # {SENTINELS["patch"]}
diff --git a/tests/{SENTINELS["pr_test"]}.py b/tests/{SENTINELS["pr_test"]}.py
--- /dev/null
+++ b/tests/{SENTINELS["pr_test"]}.py
@@ -0,0 +1,2 @@
+def test_pr():
+    assert True
"""
ISSUE = "render() crashes on an empty list; it should return an empty string."
SOURCE = "src/widgets/render.py"
OWN_TEST = "tests/test_receipts_fix.py"
OWN_TEXT = "from widgets.render import render\n\n\ndef test_empty():\n    assert render([]) == ''\n"
EXISTING = "tests/test_render.py"
EXISTING_TEXT = "from widgets.render import render\n\n\ndef test_a():\n    assert render(['a']) == 'a'\n"


def diff_of(*paths: str, pad: int = 0) -> str:
    out = []
    for path in paths:
        out.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y{'z' * pad}\n")
    return "".join(out)


GOOD_DIFF = diff_of(SOURCE, OWN_TEST)


def target(**more: object) -> Target:
    fields: dict[str, object] = {
        "id": "verified~octo__widgets-1",
        "tier": Tier.A,
        "repo": "octo/widgets",
        "base_sha": "a" * 40,
        "patch": AGENT_PATCH,
        "title": "Fix render on empty input",
        "body": f"This PR fixes render. {SENTINELS['body']}",
        "issue": ISSUE,
        "commit_messages": SENTINELS["commits"],
        "session_summary": SENTINELS["summary"],
    }
    return Target.model_validate(fields | more)


CLAIM = Claim(
    id="fixed_x",
    type=ClaimType.FIXED_X,
    text=f"fixed it {SENTINELS['claim_text']}",
    span=Span(source=SpanSource.PR_TITLE, text="Fix render on empty input"),
)


def answer(text: str, micro_usd: int = 1_000) -> ModelResponse:
    return ModelResponse(
        model=SURROGATE_MODEL,
        text=text,
        finish_reason="stop",
        usage=Usage(input_tokens=100, output_tokens=20),
        cost=StageCost(stage="x", model=SURROGATE_MODEL, input_tokens=100, output_tokens=20, micro_usd=micro_usd),
    )


def submission(sources: Sequence[str] = (SOURCE,), tests: Sequence[str] = (OWN_TEST,)) -> str:
    body = json.dumps({"source_files": list(sources), "test_files": list(tests)})
    return f"Done.\n```json\n{body}\n```"


class Client:
    """Answers every surrogate with one command, then the submission; each answer priced under its request's stage."""

    def __init__(self, text: Callable[[int], str] | None = None, fail: BaseException | None = None) -> None:
        self.requests: list[ModelRequest] = []
        self.text = text or (lambda n: "Look.\n```bash\nls\n```" if n % 2 == 0 else submission())
        self.fail = fail

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        got = answer(self.text(len(self.requests) - 1))
        assert got.cost is not None
        return got.model_copy(update={"cost": got.cost.model_copy(update={"stage": request.stage})})


class Runner:
    kind = RunnerKind.PYTEST

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["python", "-m", "pytest", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


def run(side: Side, outcome: Outcome, failure: FailureType | None = None, *, fresh: bool = True) -> RunResult:
    return RunResult(side=side, outcome=outcome, failure_type=failure, exit_code=0, fresh_container=fresh, duration_s=1)


# (side, tests, files) -> the run_test result
TestScript = Callable[[Side, tuple[str, ...], Mapping[str, str]], RunResult]


def fixes_it(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
    """Its own test fails on base on an assertion and passes on the surrogate; existing tests pass on the surrogate."""
    if side is Side.BASE:
        return run(side, Outcome.FAIL, FailureType.ASSERTION)
    return run(side, Outcome.PASS)


@dataclass
class Box:
    """A WorkspaceReader: loop commands answer ok, the diff command reports `diff`'s size, and copy_out_text reads
    `diff` or the files the fixer wrote."""

    diff: str = GOOD_DIFF
    tests: TestScript = fixes_it
    reported_size: int | None = None
    base_cases: tuple[CaseResult, ...] = (
        CaseResult(name=f"{EXISTING}::test_a", outcome=Outcome.PASS),
        CaseResult(name=f"{EXISTING}::test_b", outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION),
    )
    written: dict[str, str] = field(default_factory=lambda: {OWN_TEST: OWN_TEXT})
    tree: dict[str, str] = field(default_factory=lambda: {SOURCE: "def render(items): ...\n", EXISTING: EXISTING_TEXT})
    prepare_error: BaseException | None = None
    prepared: list[tuple[Side, str | None]] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    test_calls: list[dict[str, Any]] = field(default_factory=list)
    case_calls: list[dict[str, Any]] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        if self.prepare_error is not None:
            raise self.prepare_error
        self.prepared.append((side, patch))
        return Workspace(id=f"ws-{len(self.prepared)}", side=side, root="/testbed", image_digest="sha256:i")

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        self.commands.append(list(command))
        if len(command) > 2 and command[2] == DIFF_SCRIPT:
            size = self.reported_size if self.reported_size is not None else len(self.diff.encode())
            return run(workspace.side, Outcome.PASS, fresh=False).model_copy(update={"stdout_tail": f"{size}\n"})
        return run(workspace.side, Outcome.PASS, fresh=False)

    def run_test(self, workspace: Workspace, tests: Sequence[str], **options: Any) -> RunResult:
        self.test_calls.append({"side": workspace.side, "tests": tuple(tests), **options})
        assert options["fresh"] is True and options["seed"] == 0
        return self.tests(workspace.side, tuple(tests), options.get("files") or {})

    def run_cases(self, workspace: Workspace, tests: Sequence[str], **options: Any) -> SuiteRun:
        self.case_calls.append({"side": workspace.side, "tests": tuple(tests), **options})
        return SuiteRun(result=run(workspace.side, Outcome.FAIL, FailureType.ASSERTION), cases=self.base_cases)

    def tree_files(self, workspace: Workspace) -> tuple[str, ...]:
        return tuple(sorted(self.tree))

    def read_text(self, workspace: Workspace, path: str, *, max_bytes: int = 4 * 1024 * 1024) -> str:
        return self.tree[path]

    def copy_out_text(self, workspace: Workspace, rel_path: str, *, max_bytes: int = 8 * 1024 * 1024) -> str:
        if rel_path == DIFF_PATH:
            return self.diff
        return self.written[rel_path]

    def close(self, workspace: Workspace) -> None:
        self.closed.append(workspace.id)


def build(
    box: Box, client: Client | None = None, *, k: int = 1, the_target: Target | None = None
) -> tuple[tuple[Surrogate, ...], Client]:
    client = client or Client()
    builder: SurrogateBuilder = LoopSurrogateBuilder(client, clock=lambda: 0.0)
    made = builder.build(the_target or target(), CLAIM, sandbox=box, runner=Runner(), k=k)
    return made, client


def test_missing_git_stops_surrogates_before_model_calls() -> None:
    class NoGit(Box):
        def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
            if tuple(command) == GIT_PREFLIGHT:
                return run(workspace.side, Outcome.ERROR).model_copy(update={"exit_code": 127})
            return super().run(workspace, command, **options)

    box = NoGit()
    made, client = build(box)
    assert len(made) == 1 and not made[0].plausible and "git" in made[0].why
    assert not client.requests and sum(cost.micro_usd for cost in made[0].costs) == 0
    assert len(box.closed) == 1


def test_git_at_another_path_on_the_diff_shell_path_still_builds_surrogates(tmp_path: Path) -> None:
    git = tmp_path / "image-bin" / "git"
    git.parent.mkdir()
    git.write_text("#!/bin/sh\nprintf 'alternate image git\\n'\n")
    git.chmod(0o755)

    class AlternateGit(Box):
        def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
            if tuple(command) == GIT_PREFLIGHT:
                # Model the image's login-shell PATH without loading host startup files.
                if tuple(command[: len(SHELL)]) == tuple(SHELL):
                    actual = ["/bin/bash", "--noprofile", "--norc", "-c", command[-1]]
                else:
                    # An absolute image path has no corresponding executable here.
                    actual = [str(tmp_path / "image-root" / command[0].lstrip("/")), *command[1:]]
                try:
                    done = subprocess.run(actual, env={"PATH": str(git.parent)}, capture_output=True, text=True)
                except FileNotFoundError:
                    return run(workspace.side, Outcome.ERROR).model_copy(update={"exit_code": 127})
                assert done.returncode == 0 and done.stdout.strip() == "alternate image git"
            return super().run(workspace, command, **options)

    made, client = build(AlternateGit())
    assert len(made) == 1 and made[0].plausible, made[0].why
    assert client.requests


def test_a_fix_whose_own_test_fails_on_base_and_passes_on_it_with_existing_tests_passing_is_plausible() -> None:
    box = Box()
    made, client = build(box, k=3)
    assert [s.id for s in made] == ["fixed_x.s1", "fixed_x.s2", "fixed_x.s3"]
    assert all(s.plausible for s in made), [s.why for s in made]
    assert made[0].patch == GOOD_DIFF  # from the workspace, never the answer
    assert sorted({r.temperature for r in client.requests}) == list(TEMPERATURES)
    assert {(r.model, r.stage) for r in client.requests} == {(SURROGATE_MODEL, SURROGATE_STAGE)}
    assert [c.stage for c in made[0].costs] == [SURROGATE_STAGE, SURROGATE_STAGE]
    assert made[0].costs[0].micro_usd == 2_000 and made[0].costs[1].model is None  # model calls, then sandbox time
    # Each surrogate: a fresh base workspace for the fixer, then base with its patch, both closed.
    assert box.prepared == [(Side.BASE, None), (Side.SURROGATE, GOOD_DIFF)] * 3
    assert len(box.closed) == 6
    # The diff command names the submitted files after the script, never inside it.
    diff_command = next(c for c in box.commands if len(c) > 2 and c[2] == DIFF_SCRIPT)
    assert diff_command[:2] == ["bash", "-lc"] and diff_command[4:] == [SOURCE, OWN_TEST]
    # (a) the existing test over the changed source, on base, then on the surrogate pointed at its base passes only.
    assert [c["tests"] for c in box.case_calls[:1]] == [(EXISTING,)]
    first = box.test_calls[:3]
    assert (first[0]["side"], first[0]["tests"], first[0]["files"]) == (Side.BASE, (OWN_TEST,), {OWN_TEST: OWN_TEXT})
    assert (first[1]["side"], first[1]["tests"], first[1]["expect"]) == (
        Side.SURROGATE,
        (EXISTING,),
        [f"{EXISTING}::test_a"],
    )
    assert (first[2]["side"], first[2]["tests"], first[2]["expect"]) == (
        Side.SURROGATE,
        (OWN_TEST,),
        [f"{OWN_TEST}::test_empty"],
    )


def test_a_surrogate_that_breaks_a_base_passing_existing_test_is_not_plausible() -> None:
    def breaks(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
        if side is Side.SURROGATE and tests == (EXISTING,):
            return run(side, Outcome.FAIL, FailureType.ASSERTION)
        return fixes_it(side, tests, files)

    (made,), _ = build(Box(tests=breaks))
    assert not made.plausible and made.patch == GOOD_DIFF
    assert "existing tests that pass on base do not all pass on it" in made.why


def test_an_edited_existing_test_cannot_hide_what_the_fix_breaks() -> None:
    # The fixer breaks test_a and edits tests/test_render.py so its own copy passes. (a) keeps the edited file and
    # runs it as base has it, on base and on the surrogate, so the break shows.
    edited = EXISTING_TEXT.replace("== 'a'", "== 'a,'")

    def hides(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
        if side is Side.SURROGATE and tests == (EXISTING,) and files.get(EXISTING) == EXISTING_TEXT:
            return run(side, Outcome.FAIL, FailureType.ASSERTION)  # the base text of the test fails on the fix
        return fixes_it(side, tests, files)

    box = Box(diff=diff_of(SOURCE, EXISTING, OWN_TEST), tests=hides, written={OWN_TEST: OWN_TEXT, EXISTING: edited})
    (made,), _ = build(box)
    assert not made.plausible
    assert "existing tests that pass on base do not all pass on it" in made.why
    assert [(c["tests"], c["files"]) for c in box.case_calls] == [((EXISTING,), {EXISTING: EXISTING_TEXT})]
    on_it = [c for c in box.test_calls if c["side"] is Side.SURROGATE and c["tests"] == (EXISTING,)]
    assert [c["files"] for c in on_it] == [{EXISTING: EXISTING_TEXT}]


def test_a_surrogate_whose_own_test_fails_on_it_is_not_plausible() -> None:
    def own_fails(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
        if side is Side.SURROGATE and tests == (OWN_TEST,):
            return run(side, Outcome.FAIL, FailureType.ASSERTION)
        return fixes_it(side, tests, files)

    (made,), _ = build(Box(tests=own_fails))
    assert not made.plausible and "its own tests do not all pass on it" in made.why


@pytest.mark.parametrize(
    "on_base",
    [
        run(Side.BASE, Outcome.PASS),
        run(Side.BASE, Outcome.FAIL, FailureType.IMPORT),
        run(Side.BASE, Outcome.FAIL, FailureType.CRASH_ELSEWHERE),
        run(Side.BASE, Outcome.ERROR),
    ],
    ids=["passes", "import", "crash-elsewhere", "error"],
)
def test_the_base_fail_rule_a_surrogate_none_of_whose_tests_fails_on_base_on_a_counted_failure_is_not_plausible(
    on_base: RunResult,
) -> None:
    def script(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
        return on_base if side is Side.BASE else run(side, Outcome.PASS)

    box = Box(tests=script)
    (made,), _ = build(box)
    assert not made.plausible
    assert "none of its own tests fails on base on a counted failure" in made.why
    assert all(side is not Side.SURROGATE for side, _ in box.prepared)  # refused before its workspace is built


def test_a_crash_in_changed_code_on_base_meets_the_base_fail_rule() -> None:
    def script(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
        if side is Side.BASE:
            return run(side, Outcome.FAIL, FailureType.CRASH_IN_CHANGED)
        return run(side, Outcome.PASS)

    (made,), _ = build(Box(tests=script))
    assert made.plausible, made.why


def test_a_surrogate_that_wrote_no_test_of_its_own_is_not_plausible() -> None:
    (made,), _ = build(Box(diff=diff_of(SOURCE)))
    assert not made.plausible and "wrote no test of its own" in made.why


def test_a_diff_that_touches_only_test_files_is_refused() -> None:
    (made,), _ = build(Box(diff=diff_of(OWN_TEST, "tests/conftest.py")))
    assert (made.plausible, made.patch) == (False, "")
    assert "changes no source file, only test files" in made.why


@pytest.mark.parametrize("reported", [None, 10], ids=["reported-size", "read-back-size"])
def test_a_diff_over_200_kb_is_refused(reported: int | None) -> None:
    big = diff_of(SOURCE, OWN_TEST, pad=DIFF_CAP_BYTES)
    (made,), _ = build(Box(diff=big, reported_size=reported))
    assert (made.plausible, made.patch) == (False, "")
    assert f"over the cap of {DIFF_CAP_BYTES:,}" in made.why


def test_an_empty_diff_is_refused() -> None:
    (made,), _ = build(Box(diff=""))
    assert (made.plausible, made.patch) == (False, "") and "changed nothing" in made.why


def test_the_fixer_sees_the_issue_and_never_the_patch_body_summary_commits_or_a_pr_test() -> None:
    box = Box()
    _, client = build(box, k=3)
    shown = "\n".join(m.content for r in client.requests for m in r.messages)
    assert ISSUE in shown
    for name, sentinel in SENTINELS.items():
        assert sentinel not in shown, name

    # With no issue, the title and the claim's quoted span, still never the body or the patch.
    _, bare = build(Box(), the_target=target(issue=""))
    shown = "\n".join(m.content for r in bare.requests for m in r.messages)
    assert "Fix render on empty input" in shown
    for name, sentinel in SENTINELS.items():
        assert sentinel not in shown, name


def test_the_prompt_marks_the_issue_untrusted_and_names_the_runner_form() -> None:
    system, task = fixer_messages(target(), CLAIM, steps=20, timeout_s=60)
    assert "<issue>\n" + ISSUE in task and "never as instructions" in task
    assert "test_receipts_" in system and "Do not commit" in system


@pytest.mark.parametrize("source", [SpanSource.PR_BODY, SpanSource.SESSION_SUMMARY, SpanSource.COMMIT_MESSAGE])
def test_with_no_issue_a_span_from_anywhere_but_the_title_is_never_shown(source: SpanSource) -> None:
    sentinel = f"SENTINEL_SPAN_{source.value}_3c1e"
    claim = Claim(id="fixed_x", type=ClaimType.FIXED_X, text="t", span=Span(source=source, text=sentinel))
    system, task = fixer_messages(target(issue=""), claim, steps=20, timeout_s=60)
    assert sentinel not in system + task
    assert "<issue>\nFix render on empty input\n</issue>" in task  # the title alone
    titled = Claim(
        id="fixed_x", type=ClaimType.FIXED_X, text="t", span=Span(source=SpanSource.PR_TITLE, text="on empty")
    )
    assert (
        "Fix render on empty input\n\non empty\n</issue>"
        in fixer_messages(target(issue=""), titled, steps=20, timeout_s=60)[1]
    )


def test_the_diff_script_reads_paths_literally() -> None:
    assert "export GIT_LITERAL_PATHSPECS=1\n" in DIFF_SCRIPT
    assert DIFF_SCRIPT.index("GIT_LITERAL_PATHSPECS") < DIFF_SCRIPT.index("git add -N")


def test_a_sandbox_that_cannot_read_its_workspace_back_builds_no_surrogate_and_says_why() -> None:
    class Plain:
        def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
            raise AssertionError("never prepared")

        def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
            raise AssertionError

        def run_test(self, workspace: Workspace, tests: Sequence[str], **options: Any) -> RunResult:
            raise AssertionError

        def close(self, workspace: Workspace) -> None: ...

    client = Client()
    made = LoopSurrogateBuilder(client).build(target(), CLAIM, sandbox=Plain(), runner=Runner(), k=3)  # type: ignore[arg-type]
    assert [(s.plausible, s.patch) for s in made] == [(False, "")]
    assert "cannot read a workspace's files back" in made[0].why
    assert client.requests == []
    assert isinstance(Box(), WorkspaceReader) and not isinstance(Plain(), WorkspaceReader)


def test_a_model_or_sandbox_failure_never_raises_and_keeps_its_reason_and_spend() -> None:
    spent = StageCost(stage=SURROGATE_STAGE, model=SURROGATE_MODEL, micro_usd=55)
    (made,), _ = build(Box(), Client(fail=ModelCallError("down", UncheckedReason.RUNTIME, cost=spent)))
    assert (made.plausible, made.patch) == (False, "") and "did not submit" in made.why
    assert sum(c.micro_usd for c in made.costs) == 55

    (no_env,), _ = build(Box(prepare_error=RuntimeError("no image")))
    assert not no_env.plausible and "base workspace could not be prepared" in no_env.why

    def raising(side: Side, tests: tuple[str, ...], files: Mapping[str, str]) -> RunResult:
        raise RuntimeError("docker died")

    (broken,), _ = build(Box(tests=raising))
    assert not broken.plausible and "raised RuntimeError" in broken.why


def test_a_submission_with_a_path_outside_the_repository_is_refused_and_the_loop_goes_on() -> None:
    texts = [submission(sources=("../etc/passwd",)), submission()]
    client = Client(text=lambda n: texts[n])
    (made,), _ = build(Box(), client)
    assert made.plausible, made.why
    assert "not accepted" in client.requests[1].messages[-1].content


def test_each_surrogate_is_held_to_the_cap_the_skeptic_counts_on() -> None:
    # The skeptic builds as many surrogates as an arm's allowance covers at SURROGATE_CAP_MICRO_USD each.
    assert SURROGATE_CAP_MICRO_USD == ATTACKER_CAP_MICRO_USD == 250_000
    assert LoopSurrogateBuilder(Client())._config.cost_limit_micro_usd == SURROGATE_CAP_MICRO_USD
