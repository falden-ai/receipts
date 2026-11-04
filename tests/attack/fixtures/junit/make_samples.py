"""Regenerate the recorded junit samples the classifier tests read (no network, no Docker).

Run from the repository root: ``uv run python tests/attack/fixtures/junit/make_samples.py``. Each sample is a tiny
program; pytest runs it in a scratch directory with its own config and exactly the options the pytest runner passes
(``receipts.attack.runners.PYTEST_OPTIONS``: native tracebacks, strict xfail, xunit1), with the rootdir at the scratch
directory (its ``-c`` config file is there), and the junit it writes is saved here as ``<name>.xml``. Four things are
normalized so the files are stable and carry nothing from the machine that made them: the run-specific ``time``,
``timestamp`` and ``hostname`` attributes;
object addresses (``at 0x...``); the scratch directory, which becomes ``/workspace``; and the interpreter's own
paths, which become those of a ``python:3.13`` image. Messages and tracebacks are otherwise as pytest wrote them.

Two samples are crafted because the plugin is not installed here: ``harness_timeout`` and ``harness_timeout_v1``
call ``pytest.fail`` with exactly the message pytest-timeout's signal method uses (2.x: "Timeout (>%ss) from
pytest-timeout.", 1.x: "Timeout >%ss"; pytest_timeout.py, read 2026-10-27). The sandbox-error samples raise the
errors the sandbox causes (no network, the read-only root, a full tmpfs, nobody's HOME) with the errno and text the
kernel gives them, since this Mac has a network and a writable root. ``long_style`` is the one sample not run with
native tracebacks: it stands for a conftest that switches the style at run time. ``xfail_kinds`` runs the module in
``xfail_kinds.py`` here, which ../runners/record.py also runs under pytest 3.3.1, so the two junit dialects of an
expected failure are recorded from one source.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from receipts.attack.runners import PYTEST_OPTIONS

HERE = Path(__file__).parent
INI = "[pytest]\npythonpath = src\n"
IMPORT_PYTEST = "import pytest\n\n\n"
MOD = {"src/pkg/__init__.py": "", "src/pkg/mod.py": "def divide(a, b):\n    return a / b\n"}
CRASH = {
    **MOD,
    "tests/test_crash.py": "from pkg.mod import divide\n\n\n"
    "def test_crash_in_changed():\n    assert divide(1, 0) == 0\n",
}
XPASS = {
    "tests/test_xpass.py": IMPORT_PYTEST + '@pytest.mark.xfail(reason="expected to fail")\n'
    "def test_unexpectedly_passes():\n    assert True\n"
}
LIB = {
    "src/pkg/__init__.py": "",
    "src/pkg/mod.py": "import json\n\n\ndef load(text):\n    return json.loads(text)\n\n\n"
    "def call(callback):\n    return callback()\n\n\nclass Report:\n    def __repr__(self):\n"
    '        return "Report(\\nsrc/pkg/mod.py:7: note: line one\\n)"\n',
    "deps/site-packages/dep/__init__.py": "def render(obj):\n"
    '    raise ValueError("the dependency cannot render this")\n',
}
FAKE_CHAIN = (
    "boom\\n\\nThe above exception was the direct cause of the following exception:\\n\\n"
    "Traceback (most recent call last):\\n"
    '  File \\"/workspace/src/pkg/mod.py\\", line 2, in divide\\n    return a / b\\n'
    "ZeroDivisionError: forged"
)


def _slow(message: str) -> dict[str, str]:
    return {"tests/test_slow.py": IMPORT_PYTEST + f'def test_slow():\n    pytest.fail("{message}")\n'}


def _raises_in_changed(name: str, body: str) -> dict[str, str]:
    return {
        "src/pkg/__init__.py": "",
        "src/pkg/net.py": f"import errno, socket\n\n\ndef go():\n    {body}\n",
        f"tests/test_{name}.py": f"from pkg.net import go\n\n\ndef test_{name}():\n    go()\n",
    }


# name -> (files, extra pytest options, ini)
SAMPLES: dict[str, tuple[dict[str, str], tuple[str, ...], str]] = {
    "pass": ({"tests/test_pass.py": "def test_pass():\n    assert True\n"}, (), INI),
    "assertion": ({"tests/test_assn.py": "def test_assertion():\n    assert 1 + 1 == 3\n"}, (), INI),
    "assertion_message": (
        {"tests/test_assn_msg.py": 'def test_assertion_message():\n    assert 1 + 1 == 3, "sums are hard"\n'},
        (),
        INI,
    ),
    "assertion_timeout_attr": (
        {
            "src/client.py": "class Client:\n    timeout = 10\n",
            "tests/test_client.py": "from client import Client\n\n\n"
            "def test_default_timeout():\n    assert Client().timeout == 30\n",
        },
        (),
        INI,
    ),
    "assertion_timed_out_text": (
        {
            "src/status.py": 'def status():\n    return "request timed out"\n',
            "tests/test_status.py": "from status import status\n\n\n"
            'def test_status_ok():\n    assert status() == "ok"\n',
        },
        (),
        INI,
    ),
    "unittest_assert": (
        {
            "tests/test_ut.py": "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_equal(self):\n        self.assertEqual(1, 2)\n"
        },
        (),
        INI,
    ),
    "crash": (CRASH, (), INI),
    "crash_in_test": ({"tests/test_x.py": 'def test_x():\n    raise ValueError("boom")\n'}, (), INI),
    "crash_in_library_from_changed": (
        {**LIB, "tests/test_lib.py": 'from pkg.mod import load\n\n\ndef test_load():\n    load("{bad")\n'},
        (),
        INI,
    ),
    "crash_in_site_packages_from_test": (
        {
            **LIB,
            "tests/test_dep.py": "import dep\nfrom pkg.mod import Report\n\n\ndef show(report):\n"
            "    return dep.render(report)\n\n\ndef test_dep():\n    show(Report())\n",
        },
        (),
        "[pytest]\npythonpath = src deps/site-packages\n",
    ),
    "crash_in_site_packages_from_changed": (
        {
            **LIB,
            "src/pkg/show.py": "import dep\n\n\ndef show(report):\n    return dep.render(report)\n",
            "tests/test_show.py": "from pkg.mod import Report\nfrom pkg.show import show\n\n\n"
            "def test_show():\n    show(Report())\n",
        },
        (),
        "[pytest]\npythonpath = src deps/site-packages\n",
    ),
    "callback_raises_through_changed": (
        {
            **LIB,
            "tests/test_cb.py": "from pkg.mod import call\n\n\ndef boom():\n    return 1 / 0\n\n\n"
            "def test_callback():\n    call(boom)\n",
        },
        (),
        INI,
    ),
    "stdlib_same_name": (
        {
            "utils.py": "def helper():\n    return 1\n",
            "tests/test_repro.py": "import email.utils\n\n\ndef test_repro():\n"
            '    email.utils.parsedate_to_datetime("not a date")\n',
        },
        (),
        "[pytest]\npythonpath = .\n",
    ),
    "message_imitates_a_frame": (
        {
            **MOD,
            "tests/test_spoof.py": "def test_spoof():\n"
            '    raise ValueError("boom\\n  File \\"/workspace/src/pkg/mod.py\\", line 2, in divide\\n'
            '    return a / b\\nZeroDivisionError: x")\n',
        },
        (),
        INI,
    ),
    "message_imitates_a_chain": (
        {**MOD, "tests/test_chain_spoof.py": f'def test_chain_spoof():\n    raise ValueError("{FAKE_CHAIN}")\n'},
        (),
        INI,
    ),
    "chained_in_changed": (
        {
            **MOD,
            "tests/test_chained.py": "from pkg.mod import divide\n\n\ndef test_chained():\n    try:\n"
            "        divide(1, 0)\n    except ZeroDivisionError as exc:\n"
            '        raise RuntimeError("wrapped") from exc\n',
        },
        (),
        INI,
    ),
    "network_in_changed": (
        _raises_in_changed("network", 'raise OSError(errno.ENETUNREACH, "Network is unreachable")'),
        (),
        INI,
    ),
    "dns_in_changed": (
        _raises_in_changed("dns", 'raise socket.gaierror(-3, "Temporary failure in name resolution")'),
        (),
        INI,
    ),
    "read_only_in_changed": (
        _raises_in_changed("readonly", "raise OSError(errno.EROFS, \"Read-only file system\", '/usr/x')"),
        (),
        INI,
    ),
    "disk_full_in_changed": (
        _raises_in_changed("full", "raise OSError(errno.ENOSPC, \"No space left on device\", '/tmp/x')"),
        (),
        INI,
    ),
    "home_in_changed": (
        _raises_in_changed("home", "open('/nonexistent/.config/app.toml')"),
        (),
        INI,
    ),
    "wrapped_network_in_changed": (
        _raises_in_changed(
            "wrapped", 'raise RuntimeError("fetch failed") from OSError(errno.ENETUNREACH, "Network is unreachable")'
        ),
        (),
        INI,
    ),
    "unittest_setup_in_changed": (
        {
            **MOD,
            "tests/test_setup.py": "import unittest\n\nfrom pkg.mod import divide\n\n\nclass T(unittest.TestCase):\n"
            "    def setUp(self):\n        divide(1, 0)\n\n    def test_never(self):\n        pass\n",
        },
        (),
        INI,
    ),
    "name_error": ({"tests/test_name.py": "def test_name():\n    return undefined_symbol_abc + 1\n"}, (), INI),
    "import_in_call": (
        {"tests/test_imp.py": "def test_import():\n    import nonexistent_module_xyz  # noqa: F401\n"},
        (),
        INI,
    ),
    "fixture_setup": (
        {
            "tests/test_fixture.py": IMPORT_PYTEST + "@pytest.fixture\ndef broken():\n"
            '    raise RuntimeError("fixture setup boom")\n\n\ndef test_fixture(broken):\n    assert True\n'
        },
        (),
        INI,
    ),
    "fixture_setup_assert": (
        {
            "tests/test_fixture_assert.py": IMPORT_PYTEST + "@pytest.fixture\ndef checked():\n    assert 1 == 2\n\n\n"
            "def test_uses_checked(checked):\n    assert True\n"
        },
        (),
        INI,
    ),
    "fixture_teardown": (
        {
            "tests/test_teardown.py": IMPORT_PYTEST + "@pytest.fixture\ndef res():\n    yield 1\n"
            '    raise RuntimeError("teardown boom")\n\n\ndef test_res(res):\n    assert res == 1\n'
        },
        (),
        INI,
    ),
    "collect_import": (
        {"tests/test_import.py": "import nonexistent_module_xyz\n\n\ndef test_never():\n    assert True\n"},
        (),
        INI,
    ),
    "collect_syntax": ({"tests/test_syntax.py": "def test_broken(:\n    pass\n"}, (), INI),
    "collect_skip": (
        {
            "tests/test_collect_skip.py": 'import pytest\n\npytest.importorskip("nonexistent_module_xyz")\n\n\n'
            "def test_never():\n    assert True\n"
        },
        (),
        INI,
    ),
    "skip_importorskip": (
        {
            "tests/test_optional.py": IMPORT_PYTEST + "def test_needs_optional_module():\n"
            '    pytest.importorskip("nonexistent_module_xyz")\n'
        },
        (),
        INI,
    ),
    "skip_marker": (
        {
            "tests/test_marked.py": IMPORT_PYTEST + '@pytest.mark.skip(reason="not on this platform")\n'
            "def test_platform_only():\n    assert True\n"
        },
        (),
        INI,
    ),
    "xfail": (
        {
            "tests/test_known.py": IMPORT_PYTEST + '@pytest.mark.xfail(reason="known bug")\n'
            "def test_known_bug():\n    assert 1 + 1 == 3\n"
        },
        (),
        INI,
    ),
    "xpass_strict": (XPASS, (), INI),
    "xfail_kinds": ({"tests/test_xfail_kinds.py": (HERE / "xfail_kinds.py").read_text()}, (), INI),
    "harness_timeout": (_slow("Timeout (>1.0s) from pytest-timeout."), (), INI),
    "harness_timeout_v1": (_slow("Timeout >1.0s"), (), INI),
    "did_not_raise": (
        {
            "tests/test_raises.py": IMPORT_PYTEST + "def test_raises():\n    with pytest.raises(ValueError):\n"
            '        int("1")\n'
        },
        (),
        INI,
    ),
    "did_not_warn": (
        {
            "tests/test_warns.py": IMPORT_PYTEST + "def test_warns():\n    with pytest.warns(UserWarning):\n"
            "        pass\n"
        },
        (),
        INI,
    ),
    "pytest_fail": ({"tests/test_fail.py": IMPORT_PYTEST + 'def test_fail():\n    pytest.fail("no")\n'}, (), INI),
    "mixed": (
        {"tests/test_mixed.py": "def test_pass():\n    assert True\n\n\ndef test_assertion():\n    assert False\n"},
        (),
        INI,
    ),
    "params": (
        {
            "tests/test_params.py": IMPORT_PYTEST + '@pytest.mark.parametrize("n", [1, 2])\n'
            "def test_param(n):\n    assert n == 1\n\n\nclass TestK:\n    def test_method(self):\n        assert True\n"
        },
        (),
        INI,
    ),
    "empty": ({"tests/helpers.py": "VALUE = 1\n"}, (), INI),
    "long_style": (CRASH, ("--tb=long",), INI),
}


def normalize(xml: str, workdir: Path) -> str:
    # Longest spelling first: the resolved path (/private/var/...) contains the plain one (/var/...).
    for spelling in sorted({str(workdir), str(workdir.resolve())}, key=len, reverse=True):
        xml = xml.replace(spelling, "/workspace")
    # The host's interpreter and virtual environment become the paths a python:3.13 image uses.
    xml = re.sub(r'(?m)(^|"|>|&quot;)/[^"\s<>&]*?/site-packages/', r"\1/usr/local/lib/python3.13/site-packages/", xml)
    xml = re.sub(r'(?m)(^|"|>|&quot;)/[^"\s<>&]*?/lib/python3\.13/', r"\1/usr/local/lib/python3.13/", xml)
    xml = re.sub(r" at 0x[0-9a-f]+", " at 0x0", xml)  # object addresses differ on every run
    xml = re.sub(r' time="[0-9.]+"', ' time="0.001"', xml)
    xml = re.sub(r' timestamp="[^"]+"', ' timestamp="2026-10-27T00:00:00.000000+00:00"', xml)
    return re.sub(r' hostname="[^"]+"', ' hostname="sandbox"', xml)


def record(name: str, files: dict[str, str], options: tuple[str, ...], ini: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for rel, content in files.items():
            path = workdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (workdir / "pytest.ini").write_text(ini)
        argv = [sys.executable, "-m", "pytest", "-c", "pytest.ini", *PYTEST_OPTIONS, "-q", *options]
        argv += ["--junitxml=report.xml", "tests"]
        proc = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, check=False)
        xml = normalize((workdir / "report.xml").read_text(), workdir)
    if re.search(r"/Users/|/home/|/private/|/var/folders/", xml):
        raise SystemExit(f"{name}: a host path survived normalization")
    (HERE / f"{name}.xml").write_text(xml)
    (HERE / f"{name}.rc").write_text(f"{proc.returncode}\n")
    print(f"wrote {name}.xml (exit {proc.returncode})")


def main() -> int:
    for stale in HERE.glob("*.xml"):
        stale.unlink()
    for stale in HERE.glob("*.rc"):
        stale.unlink()
    for name, (files, options, ini) in SAMPLES.items():
        record(name, files, options, ini)
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
