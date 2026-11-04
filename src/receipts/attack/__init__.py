"""Lane attack: the sandbox (RT2), the suite runners, the attackers (AT2 to AT6) and mutation adequacy (MU1).

RT2: the Docker sandbox and its run policy, the patch chain (the SWE-bench harness's own: git apply --verbose, then
git apply --verbose --reject, then GNU patch --batch --fuzz=5, each on the tree the one before left; REG-569), the
git-history strip and its assertion, the traceback reader and the run classifier, and the hostile conftest-secret
probe. The suite runners run pytest, Django and sympy tests and read their cases back. AT3 (regression), AT2
stage 1 (correctness) and AT4 (differential) are the attackers, each a contracts.Attacker implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from receipts.attack.classify import (
    JunitParseError,
    SuiteRun,
    classify_suite,
    exit_agrees,
    import_failure_of,
    junit_cases,
    reduce_cases,
    reduce_run,
    with_expect,
)
from receipts.attack.correctness import CorrectnessAttacker
from receipts.attack.differential import DifferentialAttacker
from receipts.attack.git_history import (
    HistoryStripError,
    assert_history_stripped,
    init_single_commit,
    strip_git_history,
    strip_refs_in_place,
    strip_workspace,
)
from receipts.attack.regression import RegressionAttacker
from receipts.attack.runners import (
    DjangoRunner,
    PytestRunner,
    SympyRunner,
    django_label,
    django_label_of,
    expected_cases,
    runner_for,
)
from receipts.attack.sandbox import (
    HARNESS_CHAIN,
    BuildError,
    DockerSandbox,
    EgressPolicy,
    RuntimeRefused,
    SandboxError,
    SuiteSandbox,
    apply_patch,
    check_environment,
    choose_runtime,
    docker_available,
    image_is_digest_pinned,
    require_pinned,
    runsc_available,
    truncate_tail,
)

if TYPE_CHECKING:  # mypy checks that each implementation satisfies its contract
    from receipts.contracts import Attacker, Sandbox, SuiteRunner

    _runners: tuple[SuiteRunner, ...] = (PytestRunner(), DjangoRunner(), SympyRunner())
    _attackers: tuple[Attacker, ...] = (RegressionAttacker(), CorrectnessAttacker(), DifferentialAttacker())
    _sandboxes: tuple[Sandbox, SuiteSandbox] = (DockerSandbox(), DockerSandbox())

__all__ = [
    "HARNESS_CHAIN",
    "BuildError",
    "CorrectnessAttacker",
    "DifferentialAttacker",
    "DjangoRunner",
    "DockerSandbox",
    "EgressPolicy",
    "HistoryStripError",
    "JunitParseError",
    "PytestRunner",
    "RegressionAttacker",
    "RuntimeRefused",
    "SandboxError",
    "SuiteRun",
    "SuiteSandbox",
    "SympyRunner",
    "apply_patch",
    "assert_history_stripped",
    "check_environment",
    "choose_runtime",
    "classify_suite",
    "django_label",
    "django_label_of",
    "docker_available",
    "exit_agrees",
    "expected_cases",
    "image_is_digest_pinned",
    "import_failure_of",
    "init_single_commit",
    "junit_cases",
    "reduce_cases",
    "reduce_run",
    "require_pinned",
    "runner_for",
    "runsc_available",
    "strip_git_history",
    "strip_refs_in_place",
    "strip_workspace",
    "truncate_tail",
    "with_expect",
]
