"""SK1 end to end over a scripted sandbox, one target per runner kind (pytest, Django, sympy).

The sandbox reports each run's cases under the names the real runners print, copied from the attack lane's recordings
in the SWE-bench django__django-11999 and sympy__sympy-13974 images (tests/attack/fixtures/runners in that lane): a
pytest node id, a Django label read from unittest's progress line 'test_x (app.tests.Class) ... FAIL', and sympy's
bare name from its progress line 'test_x F'. Like the attack lane's sandbox, it matches `expect` by exact name, and a
case `expect` names that the run did not report is an error. So a candidate reaches a verdict only when core names its
cases the way the runner reports them; a name core gets wrong reads as a case that never ran, and the candidate ends
unchecked."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from receipts.contracts import (
    Attack,
    CaseResult,
    CatchKind,
    Claim,
    ClaimType,
    ContextMask,
    FailureType,
    Lens,
    Outcome,
    Repro,
    RunnerKind,
    RunResult,
    Side,
    Span,
    SpanSource,
    Status,
    Target,
    Tier,
    UncheckedReason,
    Workspace,
)
from receipts.core.cases import expected_cases
from receipts.core.skeptic import TargetSkeptic

DIGEST = "sha256:" + "f" * 64


def patch_of(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        "@@ -1,2 +1,2 @@ def receipts_join(items):\n"
        '-    return "".join(items)\n'
        '+    return ",".join(items)\n'
    )


@dataclass(frozen=True)
class Kind:
    """One runner kind: its target, a repro in the runner's form, the progress lines a real run prints for it on
    the patch and on base, and how the runner names a case from one of those lines."""

    runner: RunnerKind
    repo: str
    changed: str
    test_command: str
    repro_path: str
    repro_source: str
    reads_the_changed_code: str
    patch_lines: str
    base_lines: str
    name_pattern: str  # a regex whose groups make the case name, as the runner's own parser reads it

    def target(self) -> Target:
        return Target(
            id=f"verified~{self.repo.replace('/', '__')}-1",
            tier=Tier.A,
            repo=self.repo,
            base_sha="a" * 40,
            patch=patch_of(self.changed),
            issue="joining a list puts commas between the items",
            test_runner=self.runner,
            test_command=self.test_command,
        )

    def reported(self, lines: str) -> list[tuple[str, str]]:
        """(case name, status word) for each progress line, read the way the runner's own parser reads it."""
        found: list[tuple[str, str]] = []
        for line in lines.splitlines():
            m = re.match(self.name_pattern, line)
            if m is not None:
                found.append(
                    (m.group("name") if "name" in m.groupdict() else f"{m['qual']}.{m['method']}", m["status"])
                )
        return found


KINDS = {
    "pytest": Kind(
        runner=RunnerKind.PYTEST,
        repo="octo/widgets",
        changed="src/widgets/render.py",
        test_command="pytest -rA",
        repro_path="tests/test_receipts_repro.py",
        repro_source="from widgets.render import receipts_join\n\n\ndef test_joins():\n"
        "    assert receipts_join(['a', 'b']) == 'ab'\n",
        reads_the_changed_code="def test_joins():\n    assert '\",\"' not in open('src/widgets/render.py').read()\n",
        # pytest reports node ids (the attack lane rebuilds them from its junit report, xunit1 family).
        patch_lines="tests/test_receipts_repro.py::test_joins FAILED",
        base_lines="tests/test_receipts_repro.py::test_joins PASSED",
        name_pattern=r"^(?P<name>\S+::\S+) (?P<status>PASSED|FAILED)$",
    ),
    "django": Kind(
        runner=RunnerKind.DJANGO,
        repo="django/django",
        changed="django/utils/text.py",
        test_command="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        repro_path="tests/receipts_repro/tests.py",
        repro_source="from django.test import SimpleTestCase\nfrom django.utils import text\n\n\n"
        "class ReproTests(SimpleTestCase):\n    def test_joins(self):\n"
        "        self.assertEqual(text.receipts_join(['a', 'b']), 'ab')\n",
        reads_the_changed_code="from pathlib import Path\n\nfrom django.test import SimpleTestCase\n\n\n"
        "class ReproTests(SimpleTestCase):\n    def test_joins(self):\n"
        "        self.assertNotIn('\",\"', Path('django/utils/text.py').read_text())\n",
        # unittest's verbose progress line, as in the recording: 'test_assertion (receipts_probe.tests.ProbeTests) ...'
        patch_lines="test_joins (receipts_repro.tests.ReproTests) ... FAIL",
        base_lines="test_joins (receipts_repro.tests.ReproTests) ... ok",
        name_pattern=r"^(?P<method>\w+) \((?P<qual>[\w.]+)\) \.\.\. (?P<status>ok|FAIL)$",
    ),
    "sympy": Kind(
        runner=RunnerKind.SYMPY,
        repo="sympy/sympy",
        changed="sympy/utilities/misc.py",
        test_command="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        repro_path="sympy/utilities/tests/test_receipts_repro.py",
        repro_source="from sympy.utilities import misc\n\n\ndef test_joins():\n"
        "    assert misc.receipts_join(['a', 'b']) == 'ab'\n",
        reads_the_changed_code="import linecache\n\nfrom sympy.utilities import misc\n\n\n"
        "def test_joins():\n    assert any(',' in line for line in linecache.getlines(misc.__file__))\n",
        # bin/test's progress lines, as in the recording: 'test_assertion F', 'test_pass ok'.
        patch_lines="sympy/utilities/tests/test_receipts_repro.py[1]\ntest_joins F",
        base_lines="sympy/utilities/tests/test_receipts_repro.py[1]\ntest_joins ok",
        name_pattern=r"^(?P<name>test_\w+) (?P<status>ok|F)$",
    ),
}

CLAIM = Claim(
    id="nothing_else_changed",
    type=ClaimType.NOTHING_ELSE_CHANGED,
    text="nothing else changed",
    span=Span(source=SpanSource.TEMPLATE, text="The change alters no behavior beyond what fixing the issue requires."),
)
ATTACK = Attack(
    id="regression-1",
    claim_id=CLAIM.id,
    lens=Lens.REGRESSION,
    hypothesis="joining changed",
    mask=ContextMask.DIFF,
    temperature=0.0,
)


class ScriptedRunner:
    """The target's runner as the skeptic sees it: its kind. The scripted sandbox does the reading."""

    def __init__(self, kind: RunnerKind) -> None:
        self.kind = kind

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["run", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        raise AssertionError("the scripted sandbox reads the cases itself")


@dataclass
class ScriptedSandbox:
    """Each run prints the kind's progress lines for its side; the cases are read from them by the runner's own name
    rule, and `expect` is matched by exact name, the way contracts.Sandbox.run_test promises."""

    kind: Kind
    runs: list[dict[str, Any]] = field(default_factory=list)
    prepared: list[Side] = field(default_factory=list)

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self.prepared.append(side)
        return Workspace(id=f"ws-{side.value}", side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        raise AssertionError("SK1 runs tests through run_test only")

    def run_test(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: Any,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
        expect: Sequence[str] | None = None,
    ) -> RunResult:
        assert runner.kind is self.kind.runner and fresh and expect is not None
        lines = self.kind.patch_lines if workspace.side is Side.PATCH else self.kind.base_lines
        reported = dict(self.kind.reported(lines))
        self.runs.append({"side": workspace.side, "seed": seed, "expect": tuple(expect), "reported": tuple(reported)})
        common: dict[str, Any] = {"side": workspace.side, "seed": seed, "fresh_container": True, "duration_s": 1.0}
        if any(name not in reported for name in expect):
            return RunResult(outcome=Outcome.ERROR, **common)  # an expected case that never ran: never a pass
        if any(reported[name] in ("F", "FAIL", "FAILED") for name in expect):
            return RunResult(outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION, **common)
        return RunResult(outcome=Outcome.PASS, **common)

    def close(self, workspace: Workspace) -> None:
        pass


def candidate(kind: Kind, source: str) -> Repro:
    return Repro(
        id="r1",
        attack_id=ATTACK.id,
        claim_id=CLAIM.id,
        lens=Lens.REGRESSION,
        kind=CatchKind.REGRESSION,
        files={kind.repro_path: source},
        command=("run", kind.repro_path),
    )


def judge(kind: Kind, source: str) -> tuple[Any, ScriptedSandbox]:
    box = ScriptedSandbox(kind)
    target = kind.target()
    skeptic = TargetSkeptic(target, sandbox=box, runner=ScriptedRunner(kind.runner))
    finding = skeptic.judge(
        skeptic.screen(candidate(kind, source), ATTACK),
        claim=CLAIM,
        finding_id=f"{target.id}/{CLAIM.id}/{ATTACK.id}.1",
        wall=lambda: 3.0,
    )
    skeptic.close()
    return finding, box


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
def test_a_regression_reaches_its_verdict_under_each_runner_s_own_case_names(name: str) -> None:
    kind = KINDS[name]
    expected = expected_cases(candidate(kind, kind.repro_source), kind.runner)
    assert list(expected) == [case for case, _ in kind.reported(kind.patch_lines)]  # core names = the runner's names

    finding, box = judge(kind, kind.repro_source)
    assert finding.status is Status.BLOCKER, finding.detail
    assert [(run["side"], run["seed"]) for run in box.runs] == [(Side.PATCH, s) for s in range(5)] + [
        (Side.BASE, s) for s in range(5)
    ]
    assert {run["expect"] for run in box.runs} == {expected}
    assert sorted(box.prepared) == [Side.BASE, Side.PATCH]


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
def test_a_repro_that_reads_the_changed_code_as_text_is_refuted_before_any_run(name: str) -> None:
    # Run for real, each of these fails on the patch and passes on base, because the text differs; SK1 refutes them
    # on their syntax, so no container is built for them.
    kind = KINDS[name]
    finding, box = judge(kind, kind.reads_the_changed_code)
    assert finding.status is Status.REFUTED
    assert "reflection" in (finding.detail or "")
    assert (box.runs, box.prepared) == ([], [])


def test_a_case_name_the_runner_never_reports_leaves_the_candidate_unchecked() -> None:
    # The guard this test file leans on: a name core got wrong (sympy's old '<path>::<name>') is a case that never
    # ran, so the run is an error and the candidate is unchecked, never a verdict.
    kind = KINDS["sympy"]
    box = ScriptedSandbox(kind)
    target = kind.target()

    def old_names(repro: Repro, runner: RunnerKind) -> tuple[str, ...]:
        return tuple(f"{kind.repro_path}::{case}" for case in expected_cases(repro, runner))

    skeptic = TargetSkeptic(target, sandbox=box, runner=ScriptedRunner(kind.runner), case_names=old_names)
    finding = skeptic.judge(
        skeptic.screen(candidate(kind, kind.repro_source), ATTACK),
        claim=CLAIM,
        finding_id=f"{target.id}/{CLAIM.id}/{ATTACK.id}.1",
        wall=lambda: 3.0,
    )
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.RUNTIME)
    assert len(box.runs) == 1
