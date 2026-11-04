"""Record the Django and sympy runner outputs the attack lane's parsers are built from (needs Docker and the images).

Run from the repository root, under the heavy lock, after pulling the two digest-pinned SWE-bench images below:
``uv run --frozen python tests/attack/fixtures/runners/record.py [OUT_DIR]``. Each recording runs in the image under the
run phase's own isolation (contracts.RUN_PHASE_LIMITS.run_flags with the workspace at /testbed: no network, every
capability dropped, uid 65534, a read-only root, RAM-backed /testbed and /tmp), with HOME inside /tmp, exactly as the
sandbox runs a Tier A target. The image's /testbed is copied out with ``docker cp`` (nothing runs for that) and
streamed into the /testbed tmpfs without its .git.

What is recorded, per image: a diagnostics file (user, PATH, interpreter, how the package is installed); a probe test
module written for this lane (a pass, an assertion, a crash in a "changed" library file, a crash in the test, a skip,
an expected failure, an unexpected success, a docstring, output on both streams, an import error, a name error, a
chained exception, a blocked network call, a write to the read-only root, a message that imitates a frame, and setUp
and setUpClass errors where the runner has them); and the instance's own FAIL_TO_PASS test from SWE-bench Verified,
first with only its test patch (it must fail) and then with the gold patch too (it must pass). The test and gold
patches, the .diff files here, are copied verbatim from SWE-bench Verified (public, from the projects' repositories).

The ``sympy3`` and ``django3`` modes record what the third round's parsers and the patch chain are built from: in the
sympy image, a run of two named tests in one file with the command the lane's SympyRunner builds (one ``-k``, both
names) and the same run with one ``-k`` per name (the old command, which runs only the last name); a run whose only
odd case is an unexpected pass (X); a run of test modules that import cleanly, fail to import, and raise from their
own code after and before defining a test; a run whose first module raises from its own code, which crashes sympy's
reporter; and GNU patch's own report when it applies a patch only with fuzz, and when
no fuzz applies it. In the Django image: a run whose only odd case is an unexpected success, and tests that write to
stderr with no newline before their status, or end their own output in "ok" before they skip.

The ``chain4`` and ``django4`` modes (``round4`` runs both) record what the fourth round's patch chain and import
typing are built from. ``chain4``: the SWE-bench harness's own chain (swebench 4.1.0 GIT_APPLY_CMDS) run as the
harness runs it, as root in the sympy image's own /testbed, in a control container with no network, on seven
patches (CHAIN_CASES): each command's exit code and output and the tree it left, until one exits 0, saved as
harness_chain_13974.json. ``django4``: in the Django image under the run phase's isolation, one run whose labels name
two test modules called tests that fail to import, one that imports and an app that does not exist.

The ``pytest5`` and ``django5`` modes (``round5`` runs both) record what the fifth round's expected-failure flag is
built from. ``pytest5``: pytest 3.3.1, the version SWE-bench's astropy 0.1 to 1.3 environments pin, in the Django
image's Python 3.6 under the run phase's isolation, on the module in ../junit/xfail_kinds.py, with the command the
lane's PytestRunner builds and with the same command plus ``--rootdir=.``, which the fourth round's PytestRunner
passed: each run's exit code, output and junit report, saved as pytest3_xfail.json. The image's environment has no
pytest, so pytest 3.3.1 and what it needs come from the pure-Python wheels named in PYTEST3_WHEELS, which the
recorder checks against their sha256 and puts on PYTHONPATH from /tmp; pass the directory that holds them as the
third argument. ``django5``: in the Django image, expectedFailure tests that fail as expected in the forms unittest
prints them (plain, after a docstring line, after output with no newline, a whole class marked) beside an unexpected
success, a skip and a pass (django_expected_failures.json), and a passing test whose own output is the line
"expected failure" (django_imitated_status.json).

Nothing is normalized except the run time Django and sympy print ("in 0.123s") and the time attributes of a junit
report, which become fixed values, so the recordings stay stable. Every path in them is a path inside the image.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from receipts.contracts import RUN_PHASE_LIMITS

HERE = Path(__file__).parent
DJANGO = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11999"
    "@sha256:6bf502e69a74a0a78a87565117d04369ec89fbbca71e5929110f65e3ee1f5bc6"
)
SYMPY = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.sympy__sympy-13974"
    "@sha256:49c852aecff5447405c1f9413d2d90113bd5ba58fdbd3efe6ba2f4d1f882fbc4"
)
HOME = "/tmp/home"
ENV = {"HOME": HOME, "PYTHONHASHSEED": "0"}

DIAG = r"""
id
echo "PATH=$PATH"
command -v python python3 git patch timeout || true
python --version 2>&1 || true
ls /opt/miniconda3/envs 2>/dev/null || true
ls -la /testbed | head -n 40
du -sh /testbed 2>/dev/null || true
"""
DIAG_PY = (
    "import sys, importlib; print(sys.executable, sys.version.split()[0]); print(sys.path)\n"
    "for name in ('django', 'sympy', 'pytest'):\n"
    "    try:\n"
    "        m = importlib.import_module(name); print(name, getattr(m, '__file__', None))\n"
    "    except Exception as e:\n"
    "        print(name, 'not importable', type(e).__name__)\n"
)

# ---------------------------------------------------------------- Django

DJANGO_CHANGED = "django/utils/text.py"
DJANGO_CHANGE = (
    "\n\ndef receipts_probe_crash(value):\n"
    "    # Stands in for changed code: the frame that raises is in this file.\n"
    "    return int(value)\n"
    "\n\ndef receipts_probe_library(text):\n"
    "    # Changed code that calls the standard library, which raises.\n"
    "    import json\n"
    "    return json.loads(text)\n"
)
DJANGO_PROBE = '''import socket
import sys
import unittest

from django.test import SimpleTestCase
from django.utils import text


class ProbeTests(SimpleTestCase):
    def test_pass(self):
        self.assertEqual(text.capfirst("abc"), "Abc")

    def test_assertion(self):
        self.assertEqual(text.capfirst("abc"), "xyz")

    def test_crash_in_changed(self):
        text.receipts_probe_crash("not a number")

    def test_crash_in_library_from_changed(self):
        text.receipts_probe_library("{bad json")

    def test_crash_in_test(self):
        raise KeyError("raised in the test itself")

    def test_docstring_pass(self):
        """A docstring's first line shows in the verbose output."""
        self.assertTrue(True)

    def test_docstring_fail(self):
        """A failing test with a docstring."""
        self.assertEqual(1, 2)

    @unittest.skip("not on this platform")
    def test_skipped(self):
        pass

    @unittest.expectedFailure
    def test_expected_failure(self):
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_unexpected_success(self):
        self.assertEqual(1, 1)

    def test_prints(self):
        print("printed to stdout")
        sys.stderr.write("written to stderr\\n")
        self.assertEqual(1, 1)

    def test_import_error(self):
        import receipts_no_such_module  # noqa: F401

    def test_name_error(self):
        return receipts_undefined_name  # noqa: F821

    def test_chained(self):
        try:
            text.receipts_probe_crash("x")
        except ValueError as exc:
            raise RuntimeError("wrapped") from exc

    def test_network(self):
        socket.create_connection(("192.0.2.1", 80), timeout=2)

    def test_read_only_root(self):
        open("/usr/receipts_probe", "w")

    def test_message_imitates_a_frame(self):
        raise ValueError('boom\\n  File "/testbed/django/utils/text.py", line 1, in fake\\n    fake()')

    def test_fail_raises(self):
        with self.assertRaises(ValueError):
            int("1")


class SetUpFails(SimpleTestCase):
    def setUp(self):
        text.receipts_probe_crash("in setUp")

    def test_never_runs(self):
        pass


class SetUpClassFails(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        raise RuntimeError("setUpClass boom")

    def test_never_runs(self):
        pass
'''
DJANGO_RUN = ["python", "tests/runtests.py", "--verbosity", "2", "--parallel", "1", "--settings=test_sqlite"]

# ---------------------------------------------------------------- sympy

SYMPY_CHANGED = "sympy/utilities/misc.py"
SYMPY_CHANGE = DJANGO_CHANGE
SYMPY_PROBE = """import socket
import sys

from sympy import Symbol
from sympy.utilities import misc
from sympy.utilities.pytest import XFAIL, raises, skip


def test_pass():
    assert Symbol("x") == Symbol("x")


def test_assertion():
    assert Symbol("x") == Symbol("y")


def test_crash_in_changed():
    misc.receipts_probe_crash("not a number")


def test_crash_in_library_from_changed():
    misc.receipts_probe_library("{bad json")


def test_crash_in_test():
    raise KeyError("raised in the test itself")


def test_skipped():
    skip("not on this platform")


@XFAIL
def test_expected_failure():
    assert 1 == 2


@XFAIL
def test_unexpected_success():
    assert 1 == 1


def test_prints():
    print("printed to stdout")
    sys.stderr.write("written to stderr\\n")
    assert True


def test_import_error():
    import receipts_no_such_module  # noqa: F401


def test_name_error():
    return receipts_undefined_name  # noqa: F821


def test_chained():
    try:
        misc.receipts_probe_crash("x")
    except ValueError as exc:
        raise RuntimeError("wrapped") from exc


def test_network():
    socket.create_connection(("192.0.2.1", 80), timeout=2)


def test_read_only_root():
    open("/usr/receipts_probe", "w")


def test_message_imitates_a_frame():
    raise ValueError('boom\\n  File "/testbed/sympy/utilities/misc.py", line 1, in fake\\n    fake()')


def test_raises_did_not_raise():
    raises(ValueError, lambda: int("1"))
"""
SYMPY_PROBE_PATH = "sympy/utilities/tests/test_receipts_probe.py"
SYMPY_RUN = ["python", "bin/test", "-C", "--verbose"]


def docker(*args: str, stdin: bytes | None = None, timeout: float = 1800) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout, check=False)


def must(proc: subprocess.CompletedProcess[bytes], what: str) -> bytes:
    if proc.returncode != 0:
        raise SystemExit(f"{what} failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[-2000:]}")
    return proc.stdout


def tar_of(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def stable(text: str) -> str:
    text = re.sub(r"(Ran \d+ tests? in )[0-9.]+s", r"\g<1>0.001s", text)
    text = re.sub(r"(in )[0-9.]+( seconds)", r"\g<1>0.01\g<2>", text)
    return text


class Box:
    """One run-phase container for one image, with the image's /testbed (no .git) in the /testbed tmpfs."""

    def __init__(self, image: str, scratch: Path) -> None:
        self.image = image
        tag = uuid.uuid4().hex[:10]
        source = f"receipts-rec-src-{tag}"
        must(docker("create", "--platform", "linux/amd64", "--pull=never", "--name", source, image, "true"), "create")
        try:
            self.tree = scratch / f"testbed-{tag}.tar"
            with self.tree.open("wb") as fh:
                proc = subprocess.run(["docker", "cp", f"{source}:/testbed", "-"], stdout=fh, check=False)
            if proc.returncode != 0:
                raise SystemExit("docker cp of /testbed failed")
        finally:
            docker("rm", "-f", source)
        self.name = f"receipts-rec-run-{tag}"
        flags = RUN_PHASE_LIMITS.run_flags(workspace="/testbed")
        create = ["create", "--platform", "linux/amd64", "--name", self.name, "--init", "--runtime=runc"]
        must(docker(*create, "--pull=never", *flags, "-w", "/testbed", image, "sleep", "3600"), "create run box")
        must(docker("start", self.name), "start")
        extract = ["exec", "-i", self.name, "tar", "-x", "-f", "-", "-C", "/testbed", "--strip-components=1"]
        must(docker(*extract, "--exclude=.git", stdin=self.tree.read_bytes()), "inject /testbed")
        must(docker("exec", self.name, "mkdir", "-p", HOME), "mkdir HOME")
        # SWE-bench images keep the repository's interpreter in a conda environment that a plain exec does not
        # activate, so its bin directory goes first on PATH when it exists.
        conda = "/opt/miniconda3/envs/testbed/bin"
        found = docker("exec", self.name, "test", "-x", f"{conda}/python").returncode == 0
        image_path = must(docker("exec", self.name, "sh", "-c", "echo $PATH"), "PATH").decode().strip()
        self.env = {**ENV, "PATH": f"{conda}:{image_path}" if found else image_path}

    def put(self, files: dict[str, str]) -> None:
        must(docker("exec", "-i", self.name, "tar", "-x", "-f", "-", "-C", "/testbed", stdin=tar_of(files)), "put")

    def append(self, path: str, text: str) -> None:
        current = must(docker("exec", self.name, "cat", f"/testbed/{path}"), f"read {path}").decode()
        self.put({path: current + text})

    def patch(self, diff: str) -> None:
        must(docker("exec", "-i", "-w", "/testbed", self.name, "git", "apply", "-", stdin=diff.encode()), "git apply")

    def run(self, argv: list[str], *, timeout: int = 900) -> dict[str, object]:
        env = [part for key, value in self.env.items() for part in ("-e", f"{key}={value}")]
        started = time.monotonic()
        proc = docker("exec", "-w", "/testbed", *env, self.name, "timeout", "-k", "5", str(timeout), *argv)
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "seconds": round(time.monotonic() - started, 1),
            "stdout": stable(proc.stdout.decode("utf-8", "replace")),
            "stderr": stable(proc.stderr.decode("utf-8", "replace")),
        }

    def shell(self, script: str) -> str:
        env = [part for key, value in self.env.items() for part in ("-e", f"{key}={value}")]
        proc = docker("exec", "-w", "/testbed", *env, self.name, "sh", "-c", script)
        return proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")

    def diag(self) -> str:
        python = self.run(["python", "-c", DIAG_PY])
        return self.shell(DIAG) + str(python["stdout"]) + str(python["stderr"])

    def close(self) -> None:
        docker("rm", "-f", "-v", self.name)
        self.tree.unlink(missing_ok=True)


def save(out: Path, name: str, record: dict[str, object]) -> None:
    (out / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {name}.json: exit {record['exit_code']} in {record['seconds']}s", flush=True)


def record_django(out: Path, scratch: Path) -> None:
    box = Box(DJANGO, scratch)
    try:
        (out / "django_diag.txt").write_text(box.diag())
        box.append(DJANGO_CHANGED, DJANGO_CHANGE)
        box.put({"tests/receipts_probe/__init__.py": "", "tests/receipts_probe/tests.py": DJANGO_PROBE})
        save(out, "django_probe", box.run([*DJANGO_RUN, "receipts_probe"]))
        save(out, "django_missing_label", box.run([*DJANGO_RUN, "receipts_no_such_label"]))
        box.patch((HERE / "django_11999_test.diff").read_text())
        save(out, "django_f2p_base", box.run([*DJANGO_RUN, "model_fields.tests.GetFieldDisplayTests"]))
        box.patch((HERE / "django_11999_gold.diff").read_text())
        save(out, "django_f2p_gold", box.run([*DJANGO_RUN, "model_fields.tests.GetFieldDisplayTests"]))
    finally:
        box.close()


def record_sympy(out: Path, scratch: Path) -> None:
    box = Box(SYMPY, scratch)
    try:
        (out / "sympy_diag.txt").write_text(box.diag())
        box.append(SYMPY_CHANGED, SYMPY_CHANGE)
        box.put({SYMPY_PROBE_PATH: SYMPY_PROBE})
        save(out, "sympy_probe", box.run([*SYMPY_RUN, SYMPY_PROBE_PATH]))
        save(out, "sympy_probe_k", box.run([*SYMPY_RUN, SYMPY_PROBE_PATH, "-k", "test_assertion"]))
        box.patch((HERE / "sympy_13974_test.diff").read_text())
        tensor = "sympy/physics/quantum/tests/test_tensorproduct.py"
        save(out, "sympy_f2p_base", box.run([*SYMPY_RUN, tensor]))
        box.patch((HERE / "sympy_13974_gold.diff").read_text())
        save(out, "sympy_f2p_gold", box.run([*SYMPY_RUN, tensor]))
    finally:
        box.close()


# ---------------------------------------------------------------- the third round

SYMPY_TWO = "sympy/utilities/tests/test_receipts_two.py"
SYMPY_TWO_SOURCE = """def test_two_a():
    assert 1 == 2


def test_two_b():
    assert 1 == 3


def test_two_c():
    assert True
"""
SYMPY_XPASS = "sympy/utilities/tests/test_receipts_xpass.py"
SYMPY_XPASS_SOURCE = """from sympy.utilities.pytest import XFAIL


@XFAIL
def test_passes_unexpectedly():
    assert True
"""
# Run in sorted order, a clean module first: a module whose own code raises before any file was entered crashes
# sympy's reporter itself (it has no active file yet), which SYMPY_RAISES_FIRST records.
SYMPY_MODULES = {
    "sympy/utilities/tests/test_receipts_a_ok.py": "def test_ok():\n    assert True\n",
    "sympy/utilities/tests/test_receipts_b_import_error.py": "import receipts_no_such_module  # noqa: F401\n\n\n"
    "def test_never_runs():\n    assert True\n",
    "sympy/utilities/tests/test_receipts_c_module_error.py": "def test_defined_before_the_error():\n"
    '    assert True\n\n\nraise ValueError("raised by the module\'s own code")\n',
    "sympy/utilities/tests/test_receipts_d_module_error_first.py": 'raise ValueError("raised before any test")\n\n\n'
    "def test_never_defined():\n    assert True\n",
}
SYMPY_RAISES_FIRST = {
    "sympy/utilities/tests/test_receipts_e_raises_first.py": 'raise ValueError("raised before any test")\n',
}
# GNU patch's fallback, as the SWE-bench harness runs it. The hunk changes line 10 of a 20-line file; no context line
# matches in FUZZ, and the removed line is not in the file at all in NONE (tests/attack/test_sandbox_policy.py).
LINES = "".join(f"line {i}\n" for i in range(1, 21))
_WRONG = "".join(f" wrong {i}\n" for i in (7, 8, 9))
_WRONG_AFTER = "".join(f" wrong {i}\n" for i in (11, 12, 13))
FUZZ = (
    "diff --git a/src/lines.py b/src/lines.py\n--- a/src/lines.py\n+++ b/src/lines.py\n@@ -7,7 +7,7 @@\n"
    f"{_WRONG}-line 10\n+line ten\n{_WRONG_AFTER}"
)
NONE = FUZZ.replace("-line 10\n", "-no such line\n")
GNU_PATCH = ["patch", "--batch", "--fuzz=5", "-p1", "-i"]

DJANGO_UNEXPECTED = """import unittest

from django.test import SimpleTestCase


class OnlyAnUnexpectedSuccess(SimpleTestCase):
    @unittest.expectedFailure
    def test_passes_unexpectedly(self):
        self.assertEqual(1, 1)
"""
DJANGO_NO_NEWLINE = """import sys

from django.test import SimpleTestCase


class NoNewline(SimpleTestCase):
    def test_a_writes_then_passes(self):
        sys.stderr.write("x")

    def test_b_writes_then_fails(self):
        sys.stderr.write("x")
        self.assertEqual(1, 2)

    def test_c_ends_its_output_in_ok_then_skips(self):
        sys.stderr.write("it is ok\\n")
        self.skipTest("skipped after its own output")

    def test_d_after(self):
        pass
"""


def record_sympy3(out: Path, scratch: Path) -> None:
    from receipts.attack.runners import SympyRunner
    from receipts.contracts import RunnerKind, Target, Tier

    target = Target(
        id="rec~sympy", tier=Tier.A, repo="sympy/sympy", base_sha="a" * 40, patch="x\n", test_runner=RunnerKind.SYMPY
    )
    box = Box(SYMPY, scratch)
    try:
        box.put({SYMPY_TWO: SYMPY_TWO_SOURCE, SYMPY_XPASS: SYMPY_XPASS_SOURCE, **SYMPY_MODULES, **SYMPY_RAISES_FIRST})
        named = [f"{SYMPY_TWO}::test_two_a", f"{SYMPY_TWO}::test_two_b"]
        argv = SympyRunner().command(target, named, report_path="/tmp/unused.xml")
        save(out, "sympy_k2", box.run(argv))
        save(out, "sympy_k2_one_k_per_name", box.run([*SYMPY_RUN, SYMPY_TWO, "-k", "test_two_a", "-k", "test_two_b"]))
        save(out, "sympy_xpass_only", box.run(SympyRunner().command(target, [SYMPY_XPASS], report_path="/tmp/u")))
        save(out, "sympy_modules", box.run(SympyRunner().command(target, list(SYMPY_MODULES), report_path="/tmp/u")))
        crash = SympyRunner().command(target, list(SYMPY_RAISES_FIRST), report_path="/tmp/u")
        save(out, "sympy_reporter_crash", box.run(crash))
        version = box.run(["patch", "--version"])
        record: dict[str, object] = {"patch_version": str(version["stdout"]).splitlines()[0]}
        for name, diff in (("fuzz", FUZZ), ("none", NONE)):
            box.put({"src/lines.py": LINES, f".receipts-{name}.diff": diff})
            run = box.run([*GNU_PATCH, f".receipts-{name}.diff"])
            after = box.run(["cat", "src/lines.py"])
            record[name] = {**run, "diff": diff, "file_after": after["stdout"]}
        (out / "gnu_patch_fuzz.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote gnu_patch_fuzz.json: {record['patch_version']}", flush=True)
    finally:
        box.close()


def record_django3(out: Path, scratch: Path) -> None:
    box = Box(DJANGO, scratch)
    try:
        box.put(
            {
                "tests/receipts_unexpected/__init__.py": "",
                "tests/receipts_unexpected/tests.py": DJANGO_UNEXPECTED,
                "tests/receipts_nonl/__init__.py": "",
                "tests/receipts_nonl/tests.py": DJANGO_NO_NEWLINE,
            }
        )
        save(out, "django_unexpected_success_only", box.run([*DJANGO_RUN, "receipts_unexpected"]))
        save(out, "django_no_newline", box.run([*DJANGO_RUN, "receipts_nonl"]))
    finally:
        box.close()


# ---------------------------------------------------------------- the fourth round: the harness's own patch chain

# swebench 4.1.0, harness/run_evaluation.py GIT_APPLY_CMDS, run as the harness runs them: as root, in /testbed (the
# image's own repository), each on the tree the one before left, until one exits 0.
HARNESS_CMDS = ("git apply --verbose", "git apply --verbose --reject", "patch --batch --fuzz=5 -p1 -i")
CHAIN_ROOT = "receipts_chain"
ROWS = "".join(f"row {i}\n" for i in range(1, 41))


def _chain_hunk(at: int, *, context: str = "row", old: str | None = None, new: str | None = None) -> str:
    """One hunk changing row ``at`` of a 40-row file, with three rows of context each side; ``context='wrong'``
    makes every context row one the file does not have, so only GNU patch with fuzz takes it."""
    before = "".join(f" {context} {i}\n" for i in range(at - 3, at))
    after = "".join(f" {context} {i}\n" for i in range(at + 1, at + 4))
    old = old if old is not None else f"row {at}"
    new = new if new is not None else f"row {at} changed"
    return f"@@ -{at - 3},7 +{at - 3},7 @@\n{before}-{old}\n+{new}\n{after}"


def _chain_file(case: str, name: str, *hunks: str) -> str:
    path = f"{CHAIN_ROOT}/{case}/{name}"
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n" + "".join(hunks)


CHAIN_CASES: dict[str, tuple[dict[str, str], str]] = {
    # git apply takes it as it is.
    "exact": ({"a.py": ROWS}, _chain_file("exact", "a.py", _chain_hunk(10))),
    # No context row is in the file: neither git command takes it, GNU patch does with fuzz.
    "fuzz": ({"a.py": ROWS}, _chain_file("fuzz", "a.py", _chain_hunk(10, context="wrong"))),
    # One file applies exactly and one only with fuzz: --reject applies the first and leaves b.py.rej, and GNU patch
    # then runs on that tree.
    "reject_then_gnu": (
        {"a.py": ROWS, "b.py": ROWS},
        _chain_file("reject_then_gnu", "a.py", _chain_hunk(10))
        + _chain_file("reject_then_gnu", "b.py", _chain_hunk(10, context="wrong")),
    ),
    # The file already has the hunk's new side: it looks already applied.
    "reverse": (
        {"a.py": ROWS.replace("row 10\n", "row 10 changed\n")},
        _chain_file("reverse", "a.py", _chain_hunk(10)),
    ),
    # One file, its first hunk exact and its second only with fuzz.
    "exact_then_fuzz": (
        {"a.py": ROWS},
        _chain_file("exact_then_fuzz", "a.py", _chain_hunk(10), _chain_hunk(30, context="wrong")),
    ),
    # Nothing takes it: the row it removes is in no file.
    "none": ({"a.py": ROWS}, _chain_file("none", "a.py", _chain_hunk(10, context="wrong", old="no such row"))),
    # An exact patch that lacks its last newline, as a prediction can.
    "no_newline": ({"a.py": ROWS}, _chain_file("no_newline", "a.py", _chain_hunk(10)).rstrip("\n")),
}


def _files_under(name: str, top: str) -> dict[str, str]:
    """Every file under /testbed/<top> in container ``name``, relative to it, with its text."""
    proc = docker("exec", name, "tar", "-c", "-f", "-", "-C", "/testbed", top)
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(must(proc, f"tar {top}")), mode="r") as tar:
        for member in tar:
            if member.isfile():
                handle = tar.extractfile(member)
                assert handle is not None
                files[member.name[len(top) + 1 :]] = handle.read().decode("utf-8", "replace")
    return dict(sorted(files.items()))


def record_chain4(out: Path) -> None:
    """The harness's chain, as root in the sympy image's own /testbed (a control container with no network, not the
    sandbox): for each case, each command's exit code and output and the tree it left, until one exits 0."""
    name = f"receipts-rec-chain-{uuid.uuid4().hex[:10]}"
    create = ["create", "--platform", "linux/amd64", "--pull=never", "--network=none", "--name", name]
    must(docker(*create, "-w", "/testbed", SYMPY, "sleep", "1800"), "create the harness box")
    try:
        must(docker("start", name), "start")
        git = must(docker("exec", name, "git", "--version"), "git --version").decode().strip()
        patch = must(docker("exec", name, "patch", "--version"), "patch --version").decode().splitlines()[0]
        user = must(docker("exec", name, "id", "-u"), "id").decode().strip()
        record: dict[str, object] = {"image": SYMPY, "git": git, "patch": patch, "uid": user, "cmds": HARNESS_CMDS}
        cases: dict[str, object] = {}
        for case, (files, diff) in CHAIN_CASES.items():
            top = f"{CHAIN_ROOT}/{case}"
            planted = tar_of({f"{top}/{rel}": text for rel, text in files.items()})
            must(docker("exec", "-i", name, "tar", "-x", "-f", "-", "-C", "/testbed", stdin=planted), f"put {case}")
            must(docker("exec", "-i", name, "sh", "-c", f"cat > /tmp/{case}.diff", stdin=diff.encode()), "diff")
            steps = []
            winner = None
            for cmd in HARNESS_CMDS:
                proc = docker("exec", "-w", "/testbed", name, "sh", "-c", f"{cmd} /tmp/{case}.diff")
                steps.append(
                    {
                        "cmd": cmd,
                        "exit_code": proc.returncode,
                        "stdout": proc.stdout.decode("utf-8", "replace"),
                        "stderr": proc.stderr.decode("utf-8", "replace"),
                        "tree": _files_under(name, top),
                    }
                )
                if proc.returncode == 0:
                    winner = cmd
                    break
            cases[case] = {"files": files, "diff": diff, "steps": steps, "applied_by": winner}
            print(f"chain {case}: {winner or 'nothing applied it'}", flush=True)
        record["cases"] = cases
        (out / "harness_chain_13974.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote harness_chain_13974.json: {git}; {patch}; uid {user}", flush=True)
    finally:
        docker("rm", "-f", "-v", name)


# Two apps whose tests module fails to import (a missing module, a missing name), one that imports, and an app that
# does not exist, run in one Django run: unittest prints the same "tests (unittest.loader._FailedTest)" line for both.
DJANGO_FAILED_IMPORTS = {
    "tests/receipts_brokena/__init__.py": "",
    "tests/receipts_brokena/tests.py": "import receipts_nowhere  # noqa: F401\nfrom django.test import SimpleTestCase\n"
    "\n\nclass T(SimpleTestCase):\n    def test_x(self):\n        pass\n",
    "tests/receipts_brokenb/__init__.py": "",
    "tests/receipts_brokenb/tests.py": "from django.utils.text import receipts_no_such_name  # noqa: F401\n"
    "from django.test import SimpleTestCase\n\n\nclass T(SimpleTestCase):\n    def test_y(self):\n        pass\n",
    "tests/receipts_fine/__init__.py": "",
    "tests/receipts_fine/tests.py": "from django.test import SimpleTestCase\n\n\nclass T(SimpleTestCase):\n"
    "    def test_ok(self):\n        pass\n",
}
DJANGO_FAILED_LABELS = [
    "receipts_brokena.tests",
    "receipts_brokenb.tests.T.test_y",
    "receipts_fine.tests",
    "receipts_nosuchapp.tests",
]


def record_django4(out: Path, scratch: Path) -> None:
    box = Box(DJANGO, scratch)
    try:
        box.put(DJANGO_FAILED_IMPORTS)
        save(out, "django_failed_imports", box.run([*DJANGO_RUN, *DJANGO_FAILED_LABELS]))
    finally:
        box.close()


# ---------------------------------------------------------------- the fifth round: expected failures

# pytest 3.3.1 and what it needs (py>=1.5.0, six>=1.10.0, attrs>=17.2.0, pluggy>=0.5,<0.7; setuptools is in the
# image), as the pure-Python wheels PyPI serves, each with its sha256 there. attrs 17.3.0 and pluggy 0.6.0 are the
# versions SWE-bench's astropy 1.3 environment pins beside pytest 3.3.1.
PYTEST3_WHEELS = {
    "pytest-3.3.1-py2.py3-none-any.whl": "ae4a2d0bae1098bbe938ecd6c20a526d5d47a94dc42ad7331c9ad06d0efe4962",
    "py-1.11.0-py2.py3-none-any.whl": "607c53218732647dff4acdfcd50cb62615cedf612e72d1724fb1a0cc6405b378",
    "six-1.16.0-py2.py3-none-any.whl": "8abb2f1d86890a2dfb989f9a77cfcfd3e47c2a354b01111771326f8aa26e0254",
    "attrs-17.3.0-py2.py3-none-any.whl": "e7d51b70f19a4da5fe6b3c9938983e0af3b91e230edc504bd73c443d98037063",
    "pluggy-0.6.0-py3-none-any.whl": "e160a7fcf25762bb60efc7e171d4497ff1d8d2d75a3d0df7a21b76821ecbf5c5",
}
PYTEST3_SITE = "/tmp/receipts-site"
XFAIL_KINDS = "tests/receipts_xfail/test_xfail_kinds.py"
# astropy 1.3's test_cmd in SWE-bench 4.1.0, run through the interpreter (the image has no pytest script), with the
# site directory first on PYTHONPATH as a leading assignment, the way a target's command sets its environment.
PYTEST3_COMMAND = f"PYTHONPATH={PYTEST3_SITE} python -m pytest -rA -vv -o console_output_style=classic --tb=no"


def _wheel_files(wheels: Path) -> dict[str, bytes]:
    """Every file of the pinned wheels, by its path in a site directory; any wheel with other bytes is refused."""
    import hashlib
    import zipfile

    files: dict[str, bytes] = {}
    for name, sha256 in PYTEST3_WHEELS.items():
        data = (wheels / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise SystemExit(f"{name}: not the pinned wheel")
        with zipfile.ZipFile(io.BytesIO(data)) as whl:
            for member in whl.namelist():
                if not member.endswith("/"):
                    files[member] = whl.read(member)
    return files


def _tar_of_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _stable_junit(xml: str) -> str:
    return re.sub(r' time="[0-9.]+"', ' time="0.001"', xml)


def record_pytest5(out: Path, scratch: Path, wheels: Path) -> None:
    from receipts.attack.runners import PytestRunner
    from receipts.contracts import Target, Tier

    site = _wheel_files(wheels)
    target = Target(
        id="rec~pytest3", tier=Tier.A, repo="o/r", base_sha="a" * 40, patch="x\n", test_command=PYTEST3_COMMAND
    )
    box = Box(DJANGO, scratch)
    try:
        must(docker("exec", box.name, "mkdir", "-p", PYTEST3_SITE), "mkdir the site directory")
        extract = ["exec", "-i", box.name, "tar", "-x", "-f", "-", "-C", PYTEST3_SITE]
        must(docker(*extract, stdin=_tar_of_bytes(site)), "put the wheels")
        box.put(
            {
                "tests/receipts_xfail/__init__.py": "",
                XFAIL_KINDS: (HERE.parent / "junit" / "xfail_kinds.py").read_text(),
            }
        )
        record: dict[str, object] = {"wheels": PYTEST3_WHEELS, "command": PYTEST3_COMMAND}
        for name, rootdir in (("without_rootdir", ()), ("with_rootdir", ("--rootdir=.",))):
            report = f"/tmp/receipts-{name}.xml"
            built = [a for a in PytestRunner().command(target, [XFAIL_KINDS], report_path=report) if a != "--rootdir=."]
            at = built.index(f"--junitxml={report}")
            argv = [*built[:at], *rootdir, *built[at:]]
            run = box.run(argv)
            junit = docker("exec", box.name, "cat", report)
            run["junit"] = _stable_junit(junit.stdout.decode("utf-8", "replace")) if junit.returncode == 0 else None
            record[name] = run
            print(f"pytest 3.3.1 {name}: exit {run['exit_code']}, junit {'written' if run['junit'] else 'none'}")
        version = box.run(
            ["env", f"PYTHONPATH={PYTEST3_SITE}", "python", "-c", "import pytest; print(pytest.__version__)"]
        )
        record["pytest_version"] = str(version["stdout"]).strip()
        (out / "pytest3_xfail.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote pytest3_xfail.json: pytest {record['pytest_version']}", flush=True)
    finally:
        box.close()


DJANGO_EXPECTED_FAILURES = '''import sys
import unittest

from django.test import SimpleTestCase


class ExpectedFailures(SimpleTestCase):
    @unittest.expectedFailure
    def test_a_fails_as_expected(self):
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_b_with_a_docstring(self):
        """A docstring's first line shows in the verbose output."""
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_c_writes_then_fails_as_expected(self):
        sys.stderr.write("x")
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_d_passes_unexpectedly(self):
        pass

    @unittest.skip("not here")
    def test_e_skipped(self):
        pass

    def test_f_ok(self):
        pass


@unittest.expectedFailure
class WholeClassExpected(SimpleTestCase):
    def test_g_fails_as_expected(self):
        self.assertEqual(1, 2)
'''
DJANGO_IMITATED_STATUS = """import sys

from django.test import SimpleTestCase


class Imitates(SimpleTestCase):
    def test_prints_a_status_then_passes(self):
        sys.stderr.write("expected failure\\n")

    def test_after(self):
        pass
"""


def record_django5(out: Path, scratch: Path) -> None:
    box = Box(DJANGO, scratch)
    try:
        box.put(
            {
                "tests/receipts_xf/__init__.py": "",
                "tests/receipts_xf/tests.py": DJANGO_EXPECTED_FAILURES,
                "tests/receipts_imitates/__init__.py": "",
                "tests/receipts_imitates/tests.py": DJANGO_IMITATED_STATUS,
            }
        )
        save(out, "django_expected_failures", box.run([*DJANGO_RUN, "receipts_xf"]))
        save(out, "django_imitated_status", box.run([*DJANGO_RUN, "receipts_imitates"]))
    finally:
        box.close()


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    with tempfile.TemporaryDirectory(prefix="receipts-rec-") as tmp:
        if which in ("both", "django"):
            record_django(out, Path(tmp))
        if which in ("both", "sympy"):
            record_sympy(out, Path(tmp))
        if which in ("round3", "sympy3"):
            record_sympy3(out, Path(tmp))
        if which in ("round3", "django3"):
            record_django3(out, Path(tmp))
        if which in ("round4", "chain4"):
            record_chain4(out)
        if which in ("round4", "django4"):
            record_django4(out, Path(tmp))
        if which in ("round5", "pytest5"):
            if len(sys.argv) < 4:
                raise SystemExit("pytest5 needs the directory that holds PYTEST3_WHEELS as its third argument")
            record_pytest5(out, Path(tmp), Path(sys.argv[3]))
        if which in ("round5", "django5"):
            record_django5(out, Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
