"""SK1 (b), the fake-test rules. Each rule is fed repros planted to break it, in the forms a test writes it, and clean
repros, including ones that only look like a rule (eval(repr(x)), a repository importing itself, a mock of a name the
patch changed elsewhere), must pass every rule, so a rule that fires too widely fails here too."""

from __future__ import annotations

import pytest

from receipts.core.cases import CaseRef
from receipts.core.diff import PatchFacts, patch_facts
from receipts.core.fake_tests import WHOLE_FILE, FakeRule, Scope, check_repro_files

# The patch changed render, Widget and an __init__ in widgets/render.py.
FACTS = PatchFacts(
    files=frozenset({"src/widgets/render.py"}),
    names_by_module={"widgets.render": frozenset({"render", "Widget", "__init__"})},
    packages=frozenset({"widgets"}),
)
NO_FACTS = PatchFacts(files=frozenset(), names_by_module={}, packages=frozenset())


def rules(source: str, path: str = "tests/test_repro.py", facts: PatchFacts = FACTS) -> list[FakeRule]:
    return [violation.rule for violation in check_repro_files({path: source}, facts)]


CLEAN = """\
import os
import urllib.parse
from pathlib import Path
from unittest import mock

import pytest

from widgets.render import helper, render


def test_an_empty_list_renders_empty(tmp_path):
    with mock.patch("widgets.render.helper", return_value=""):  # helper is unchanged: patching it is fine
        assert render([]) == ""
    assert urllib.parse.quote("a b") == "a%20b"
    assert not Path("relative/file.txt").exists()
    with open("/tmp/scratch.txt", "w") as handle:
        handle.write("x")
    assert os.path.exists(tmp_path)
"""


def test_a_clean_repro_breaks_no_rule() -> None:
    assert rules(CLEAN) == []


@pytest.mark.parametrize(
    "source",
    [
        "import inspect\nfrom widgets.render import render\n\n"
        "def test_x():\n    assert 'if' in inspect.getsource(render)\n",
        "from inspect import getsource as gs\n\ndef test_x():\n    assert gs(len)\n",
        "def test_x():\n    from widgets.render import render\n    assert render.__code__.co_argcount == 1\n",
        "import sys\n\ndef test_x():\n    sys.modules['widgets.render'] = object()\n",
        "import importlib, widgets\n\ndef test_x():\n    importlib.reload(widgets)\n",
        "import sys\n\ndef test_x():\n    assert sys._getframe().f_back\n",
    ],
    ids=["getsource", "aliased-import", "code-object", "sys-modules", "reload", "frames"],
)
def test_reflection_is_refused(source: str) -> None:
    assert FakeRule.REFLECTION in rules(source)


def test_running_code_is_not_reflection() -> None:
    # sympy's own idiom: a printed expression must evaluate back to the expression.
    srepr = "from sympy import Symbol, srepr\n\ndef test_x():\n    x = Symbol('x')\n    assert eval(srepr(x)) == x\n"
    run = "def test_x():\n    namespace = {}\n    exec('y = 2', namespace)\n    assert namespace['y'] == 2\n"
    assert rules(srepr) == []
    assert rules(run) == []


@pytest.mark.parametrize(
    "source",
    [
        "from unittest import mock\n\ndef test_x():\n    m = mock.Mock()\n    assert m.call_count == 1\n",
        "from unittest import mock\n\ndef test_x():\n    m = mock.Mock()\n    m.assert_called_once()\n",
        "from unittest import mock\n\ndef test_x():\n    m = mock.Mock()\n    m.assert_not_called()\n",
        "def test_x(mocker):\n    m = mocker.Mock()\n    assert m.mock_calls == []\n",
        "def test_x(mocker):\n    m = mocker.AsyncMock()\n    m.assert_awaited_once_with(1)\n",
    ],
    ids=["call_count", "assert_called_once", "assert_not_called", "mock_calls", "assert_awaited_once_with"],
)
def test_call_count_asserts_are_refused(source: str) -> None:
    assert FakeRule.CALL_COUNT in rules(source)


@pytest.mark.parametrize(
    "source",
    [
        "from unittest.mock import patch\n\n@patch('widgets.render.render')\ndef test_x(fake):\n    pass\n",
        "from unittest import mock\nimport widgets.render as w\n\n"
        "def test_x():\n    with mock.patch.object(w, 'render'):\n        pass\n",
        "def test_x(monkeypatch):\n    import widgets.render as w\n"
        "    monkeypatch.setattr(w, 'render', lambda items: '')\n",
        "def test_x(monkeypatch):\n    monkeypatch.setattr('widgets.render.render', lambda items: '')\n",
        "from widgets import render as module\n\ndef test_x():\n    module.render = lambda items: ''\n",
        "import widgets.render\n\ndef test_x():\n    setattr(widgets.render, 'Widget', object)\n",
        "def test_x(mocker):\n    mocker.patch('widgets.render.Widget')\n",
        "from unittest import mock\nimport widgets.render as w\n\n@mock.patch.multiple(w, render=mock.DEFAULT)\n"
        "def test_x(render):\n    pass\n",
        "from unittest import mock\nfrom widgets.render import Widget\n\n"
        "def test_x():\n    with mock.patch.object(Widget, 'render'):\n        pass\n",
        "from unittest import mock\n\ndef test_x(view):\n    with mock.patch.object(view, 'render'):\n        pass\n",
    ],
    ids=[
        "patch-decorator",
        "patch-object",
        "monkeypatch-object",
        "monkeypatch-string",
        "assignment",
        "setattr",
        "mocker-patch-class",
        "patch-multiple",
        "patch-object-on-a-changed-class",
        "untraced-object-by-name",
    ],
)
def test_mocks_of_the_changed_function_are_refused(source: str) -> None:
    assert FakeRule.MOCKS_CHANGED in rules(source)


@pytest.mark.parametrize(
    "source",
    [
        "from unittest.mock import patch\n\n@patch('os.render')\ndef test_x(m):\n    pass\n",
        "from unittest.mock import patch\n\n@patch('widgets.other.render')\ndef test_x(m):\n    pass\n",
        "from unittest import mock\n\n"
        "def test_x(thing):\n    with mock.patch.object(thing, '__init__'):\n        pass\n",
        "import os\n\ndef test_x(monkeypatch):\n    monkeypatch.setattr(os, 'render', None, raising=False)\n",
    ],
    ids=["another-module", "a-sibling-module", "an-untraced-dunder", "an-imported-module-the-patch-left-alone"],
)
def test_a_mock_counts_only_where_the_patch_changed_that_name(source: str) -> None:
    assert FakeRule.MOCKS_CHANGED not in rules(source)


def test_writing_over_a_changed_file_is_refused_in_any_language() -> None:
    found = check_repro_files(
        {
            "tests/test_ok.py": "def test_x():\n    pass\n",
            "src/widgets/render.py": "def render(items):\n    return ''\n",
        },
        FACTS,
    )
    assert [(v.rule, v.path) for v in found] == [(FakeRule.MOCKS_CHANGED, "src/widgets/render.py")]
    data = check_repro_files({"src/widgets/render.py": "{}"}, FACTS)
    assert [v.rule for v in data] == [FakeRule.MOCKS_CHANGED]


@pytest.mark.parametrize(
    "source",
    [
        "import socket\n\ndef test_x():\n    socket.create_connection(('example.com', 80))\n",
        "import requests\n\ndef test_x():\n    requests.get('https://example.com')\n",
        "from urllib.request import urlopen\n\ndef test_x():\n    urlopen('https://example.com')\n",
        "from urllib import request\n\ndef test_x():\n    request.urlopen('https://example.com')\n",
        "import subprocess\n\ndef test_x():\n    subprocess.run(['ls'])\n",
        "import os\n\ndef test_x():\n    os.system('ls')\n",
        "import os\n\ndef test_x():\n    os.execv('/bin/ls', ['ls'])\n",
        "def test_x():\n    assert open('/etc/passwd').read()\n",
        "from pathlib import Path\n\ndef test_x():\n    assert Path('~/.ssh/id_ed25519').exists()\n",
    ],
    ids=[
        "socket",
        "requests",
        "urlopen",
        "urllib-request",
        "subprocess",
        "os-system",
        "os-exec",
        "etc-file",
        "home-file",
    ],
)
def test_infrastructure_dependence_is_refused(source: str) -> None:
    assert FakeRule.INFRA in rules(source)


def test_the_repository_s_own_packages_are_never_the_network() -> None:
    source = "import requests\nfrom requests.adapters import HTTPAdapter\n\ndef test_x():\n    assert HTTPAdapter\n"
    patch = "diff --git a/requests/models.py b/requests/models.py\n--- a/requests/models.py\n+++ b/requests/models.py\n"
    own = patch_facts(patch, "psf/requests")
    assert own.packages == {"requests"}
    assert rules(source, facts=own) == []
    assert FakeRule.INFRA in rules(source)  # the same imports in a repro for another repository


def test_patch_facts_name_modules_without_their_source_root() -> None:
    patch = (
        "diff --git a/lib/matplotlib/axes/_base.py b/lib/matplotlib/axes/_base.py\n"
        "--- a/lib/matplotlib/axes/_base.py\n+++ b/lib/matplotlib/axes/_base.py\n"
        "@@ -1 +1 @@ def autoscale(self):\n-a\n+b\n"
    )
    facts = patch_facts(patch, "matplotlib/matplotlib")
    assert facts.names_by_module == {"matplotlib.axes._base": frozenset({"autoscale"})}
    assert facts.packages == {"matplotlib"}
    assert facts.files == {"lib/matplotlib/axes/_base.py"}


def test_a_file_that_does_not_parse_is_refused_with_its_line() -> None:
    (violation,) = check_repro_files({"tests/test_bad.py": "def test_ok():\n    pass\n\ndef broken(:\n"}, NO_FACTS)
    assert (violation.rule, violation.line) == (FakeRule.UNPARSEABLE, 4)
    assert check_repro_files({"tests/data.json": "{not python"}, NO_FACTS) == ()


def test_each_violation_names_its_file_line_and_why() -> None:
    (violation,) = check_repro_files({"tests/test_x.py": "\n\nimport socket\n"}, NO_FACTS)
    assert str(violation) == "infra at tests/test_x.py:3: imports socket: the run phase has no network"


def test_a_file_nested_past_the_parser_is_unparseable_not_a_crash() -> None:
    deep = "def test_x():\n    assert " + "(" * 100_000 + "1" + ")" * 100_000 + "\n"
    (violation,) = check_repro_files({"tests/test_deep.py": deep}, NO_FACTS)
    assert violation.rule is FakeRule.UNPARSEABLE


# ---------------------------------------------------------------- reading the changed code as text (round 3, blocker 1)

# Each repro reads src/widgets/render.py, the file the patch changed, as text. Such a test fails on the patch and
# passes on base by construction (a plain pytest run of the first one did exactly that, 5 of 5 each way, in the round
# 2 review), so each must be refused as reflection before anything runs.
READS_THE_CHANGED_CODE = {
    "open": "def test_x():\n    assert '\",\".join' not in open('src/widgets/render.py').read()\n",
    "open-keyword": "def test_x():\n    with open(file='src/widgets/render.py') as handle:\n"
    "        assert handle.read()\n",
    "io-open": "import io\n\ndef test_x():\n    assert io.open('src/widgets/render.py').read()\n",
    "builtins-open": "import builtins\n\ndef test_x():\n    assert builtins.open('src/widgets/render.py').read()\n",
    "codecs-open": "import codecs\n\ndef test_x():\n    assert codecs.open('src/widgets/render.py').read()\n",
    "tokenize-open": "import tokenize\n\ndef test_x():\n    assert tokenize.open('src/widgets/render.py').read()\n",
    "path-read_text": "from pathlib import Path\n\n"
    "def test_x():\n    assert Path('src/widgets/render.py').read_text()\n",
    "path-read_bytes": "import pathlib\n\n"
    "def test_x():\n    assert pathlib.Path('src/widgets/render.py').read_bytes()\n",
    "path-open": "from pathlib import Path as P\n\ndef test_x():\n    with P('src/widgets/render.py').open() as f:\n"
    "        assert f.read()\n",
    "path-write_text": "from pathlib import Path\n\ndef test_x():\n    Path('src/widgets/render.py').write_text('')\n",
    "module-file": "from pathlib import Path\nfrom widgets import render\n\n"
    "def test_x():\n    assert ',' in Path(render.__file__).read_text()\n",
    "linecache-getlines": "import linecache\nfrom widgets import render\n\n"
    "def test_x():\n    assert linecache.getlines(render.__file__)\n",
    "linecache-getline": "from linecache import getline\n\n"
    "def test_x():\n    assert getline('src/widgets/render.py', 2)\n",
    "module-spec": "from widgets import render\n\ndef test_x():\n    assert render.__spec__.origin\n",
    "module-loader": "from widgets import render\n\n"
    "def test_x():\n    assert render.__loader__.get_source('widgets.render')\n",
    "module-cached": "import widgets.render\n\ndef test_x():\n    assert widgets.render.__cached__\n",
    "getattr-file": "from widgets import render\n\ndef test_x():\n    assert getattr(render, '__file__')\n",
    "own-file-parents": "from pathlib import Path\n\n"
    "def test_x():\n    assert Path(__file__).parent.parent.joinpath('src', 'widgets', 'render.py').read_text()\n",
    "own-file-join": "import os\n\ndef test_x():\n"
    "    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'widgets', 'render.py')\n"
    "    assert open(path).read()\n",
    "own-file-parents-index": "from pathlib import Path\n\n"
    "def test_x():\n    assert (Path(__file__).resolve().parents[1] / 'src/widgets/render.py').read_text()\n",
    "variable": "SOURCE = 'src/widgets/render.py'\n\ndef test_x():\n    assert open(SOURCE).read()\n",
    "absolute": "def test_x():\n    assert open('/testbed/src/widgets/render.py').read()\n",
    "f-string": "NAME = 'render'\n\ndef test_x():\n    assert open(f'src/widgets/{NAME}.py').read()\n",
    "pkgutil-get_data": "import pkgutil\n\ndef test_x():\n    assert pkgutil.get_data('widgets', 'render.py')\n",
    "resources-read_text": "import importlib.resources\n\n"
    "def test_x():\n    assert importlib.resources.read_text('widgets', 'render.py')\n",
    "resources-files": "from importlib.resources import files\n\n"
    "def test_x():\n    assert (files('widgets') / 'render.py').read_text()\n",
    "resources-files-of-a-module": "from importlib.resources import files\nimport widgets\n\n"
    "def test_x():\n    assert files(widgets).joinpath('render.py').read_text()\n",
    "package-path": "import os\nimport widgets\n\n"
    "def test_x():\n    assert open(os.path.join(widgets.__path__[0], 'render.py')).read()\n",
    "find-spec-origin": "import importlib.util\n\n"
    "def test_x():\n    assert open(importlib.util.find_spec('widgets.render').origin).read()\n",
    "find-spec-of-the-package": "from importlib.util import find_spec\n\n"
    "def test_x():\n    assert find_spec('widgets')\n",
    "get-loader": "import pkgutil\n\ndef test_x():\n    assert pkgutil.get_loader('widgets.render').get_source('x')\n",
    "cwd": "from pathlib import Path\n\n"
    "def test_x():\n    assert (Path.cwd() / 'src' / 'widgets' / 'render.py').read_text()\n",
    # round 4: the round-3 review's test gaps (N3, N6 to N11), each a form the rule reads that no test covered
    "io-open_code": "import io\n\ndef test_x():\n    assert io.open_code('src/widgets/render.py').read()\n",
    "io-FileIO": "import io\n\ndef test_x():\n    assert io.FileIO('src/widgets/render.py').read()\n",
    "os-open": "import os\n\ndef test_x():\n    assert os.read(os.open('src/widgets/render.py', os.O_RDONLY), 9)\n",
    "fileinput-input": "import fileinput\n\ndef test_x():\n    assert list(fileinput.input('src/widgets/render.py'))\n",
    "fileinput-FileInput": "from fileinput import FileInput\n\n"
    "def test_x():\n    assert list(FileInput(files='src/widgets/render.py'))\n",
    "fileinput-list": "import fileinput\n\n"
    "def test_x():\n    assert list(fileinput.input(['src/widgets/render.py']))\n",
    "fileinput-tuple": "import fileinput\n\n"
    "def test_x():\n    assert list(fileinput.input(('src/widgets/render.py',)))\n",
    "shutil-copy": "import shutil\n\ndef test_x(tmp_path):\n    shutil.copy('src/widgets/render.py', tmp_path)\n",
    "shutil-copy2": "import shutil\n\ndef test_x(tmp_path):\n    shutil.copy2('src/widgets/render.py', tmp_path)\n",
    "shutil-copyfile": "import shutil\n\n"
    "def test_x(tmp_path):\n    shutil.copyfile('src/widgets/render.py', tmp_path / 'r')\n",
    "shutil-move": "import shutil\n\ndef test_x(tmp_path):\n    shutil.move('src/widgets/render.py', tmp_path)\n",
    "with_name": "from pathlib import Path\n\n"
    "def test_x():\n    assert Path('src/widgets/other.py').with_name('render.py').read_text()\n",
    "with_suffix": "from pathlib import Path\n\n"
    "def test_x():\n    assert Path('src/widgets/render.txt').with_suffix('.py').read_text()\n",
    "for-loop": "def test_x():\n    for path in ['src/widgets/render.py']:\n        assert open(path).read()\n",
    "one-of-two-values": "import os\n\ndef test_x():\n    path = 'src/widgets/render.py'\n"
    "    if os.environ.get('RECEIPTS_ELSEWHERE'):\n        path = 'notes.txt'\n    assert open(path).read()\n",
}


@pytest.mark.parametrize("source", list(READS_THE_CHANGED_CODE.values()), ids=list(READS_THE_CHANGED_CODE))
def test_reading_the_changed_code_as_text_is_reflection(source: str) -> None:
    assert FakeRule.REFLECTION in rules(source)


def test_a_repro_s_own_file_and_files_the_patch_left_alone_are_not_reflection() -> None:
    # A test may find data next to itself, and read files the patch did not change: the text is the same on both
    # sides, so it cannot fail on the patch by construction.
    data = "from pathlib import Path\n\ndef test_x():\n    assert (Path(__file__).parent / 'data.json').read_text()\n"
    other = "def test_x():\n    assert open('src/widgets/other.py').read()\n"
    temporary = "def test_x(tmp_path):\n    (tmp_path / 'render.py').write_text('x = 1')\n"
    own = "def test_x():\n    assert __file__.endswith('.py')\n"
    resource = "import pkgutil\n\ndef test_x():\n    assert pkgutil.get_data('widgets', 'data/table.json')\n"
    optional = (
        "import importlib.util\n\ndef test_x():\n    if importlib.util.find_spec('numpy') is None:\n        return\n"
    )
    for source in (data, other, temporary, own, resource, optional):
        assert rules(source) == [], source


@pytest.mark.parametrize("attribute", ["__code__", "__globals__", "__closure__", "__defaults__", "__wrapped__"])
def test_a_string_naming_a_code_attribute_is_reflection(attribute: str) -> None:
    # getattr(render, "__code__") is render.__code__ under another spelling (round-3 review, N14).
    source = f"from widgets.render import render\n\ndef test_x():\n    assert getattr(render, '{attribute}')\n"
    assert FakeRule.REFLECTION in rules(source)
    assert rules("def test_x():\n    assert 'code' and 'globals'\n") == []  # words that only look like them


def test_a_violation_names_the_changed_file_it_reads() -> None:
    (violation,) = check_repro_files({"tests/test_x.py": READS_THE_CHANGED_CODE["path-read_text"]}, FACTS)
    assert str(violation) == (
        "reflection at tests/test_x.py:4: read_text() of src/widgets/render.py, a file the patch changed"
    )


# ---------------------------------------------------------------- mocks under any name (round 3, integrator call f)

ALIASED_MOCKS = {
    "patch-as-p-decorator": "from unittest.mock import patch as p\n\n@p('widgets.render.render')\n"
    "def test_x(fake):\n    pass\n",
    "patch-as-p-context": "from unittest.mock import patch as p\n\n"
    "def test_x():\n    with p('widgets.render.Widget'):\n        pass\n",
    "patch-as-p-object": "from unittest.mock import patch as p\nimport widgets.render as w\n\n"
    "def test_x():\n    with p.object(w, 'render'):\n        pass\n",
    "mock-module-as-m": "import unittest.mock as m\n\ndef test_x():\n    with m.patch('widgets.render.render'):\n"
    "        pass\n",
    "mock-module-as-m-object": "import unittest.mock as m\nimport widgets.render as w\n\n"
    "def test_x():\n    with m.patch.object(w, 'render'):\n        pass\n",
    "mocker-patch": "def test_x(mocker):\n    mocker.patch('widgets.render.render')\n",
    "mocker-patch-object": "import widgets.render as w\n\ndef test_x(mocker):\n    mocker.patch.object(w, 'render')\n",
    "monkeypatch-setattr": "import widgets.render as w\n\n"
    "def test_x(monkeypatch):\n    monkeypatch.setattr(w, 'render', lambda items: '')\n",
    "monkeypatch-setattr-string": "def test_x(monkeypatch):\n"
    "    monkeypatch.setattr('widgets.render.render', lambda items: '')\n",
    "monkeypatch-setitem-dict": "import widgets.render as w\n\n"
    "def test_x(monkeypatch):\n    monkeypatch.setitem(w.__dict__, 'render', lambda items: '')\n",
    "namespace-dict-assignment": "import widgets.render as w\n\ndef test_x():\n    w.__dict__['render'] = None\n",
    "vars-assignment": "import widgets.render as w\n\ndef test_x():\n    vars(w)['render'] = None\n",
    "patch-dict": "from unittest import mock\nimport widgets.render as w\n\n"
    "def test_x():\n    with mock.patch.dict(w.__dict__, {'render': None}):\n        pass\n",
    "sys-modules-attribute": "import sys\n\ndef test_x():\n    sys.modules['widgets.render'].render = None\n",
    "import-module-attribute": "import importlib\n\n"
    "def test_x():\n    importlib.import_module('widgets.render').render = None\n",
}


@pytest.mark.parametrize("source", list(ALIASED_MOCKS.values()), ids=list(ALIASED_MOCKS))
def test_a_mock_of_the_changed_function_is_refused_under_any_name(source: str) -> None:
    assert FakeRule.MOCKS_CHANGED in rules(source)


@pytest.mark.parametrize(
    "source",
    [
        "from unittest.mock import patch as p\n\n@p('widgets.render.helper')\ndef test_x(m):\n    pass\n",
        "import unittest.mock as m\n\ndef test_x():\n    with m.patch('widgets.other.render'):\n        pass\n",
        "import widgets.render as w\n\ndef test_x(monkeypatch):\n    monkeypatch.setitem(w.__dict__, 'helper', 1)\n",
        "import widgets.render as w\n\ndef test_x():\n    w.__dict__['helper'] = None\n",
    ],
    ids=["aliased-unchanged-name", "aliased-sibling-module", "setitem-unchanged", "namespace-unchanged"],
)
def test_an_aliased_mock_of_something_the_patch_left_alone_is_allowed(source: str) -> None:
    assert FakeRule.MOCKS_CHANGED not in rules(source)


def test_a_path_held_by_a_name_is_read_the_same_wherever_the_name_is_met() -> None:
    # test_y meets A39 first, 10 levels inside another path, where a budget shared with the caller would run out
    # partway down the chain; a name is worked out with a budget of its own, so test_x's open(A39) still reads it.
    chain = "A0 = 'src/widgets/render.py'\n" + "".join(f"A{n} = A{n - 1}\n" for n in range(1, 40))
    deep = "X = " + " / ".join(["A39"] + ["'x'"] * 10) + "\n\n"
    tests = "def test_y():\n    open(X)\n\n\ndef test_x():\n    assert open(A39).read()\n"
    found = check_repro_files({"tests/test_x.py": chain + deep + tests}, FACTS)
    assert [(v.rule, v.line) for v in found] == [(FakeRule.REFLECTION, 48)]  # test_x's open(A39)


def test_a_path_past_path_max_names_no_file_and_cannot_grow_without_bound() -> None:
    # A path longer than PATH_MAX opens nothing, and a '+' chain through names would otherwise double its length at
    # every step. Each of these would name the changed file if the length were not capped.
    literal = f"def test_x():\n    open('/{'a/' * 2500}' + 'src/widgets/render.py')\n"
    assert rules(literal) == []
    doubling = "B0 = 'a/'\n" + "".join(f"B{n} = B{n - 1} + B{n - 1}\n" for n in range(1, 13))
    assert rules(doubling + "def test_x():\n    open('/' + B12 + 'src/widgets/render.py')\n") == []
    assert FakeRule.REFLECTION in rules("def test_x():\n    open('/' + 'a/' + 'src/widgets/render.py')\n")


# ---------------------------------------------------------------- what the rules read (round 4, Repro.cases)

SIBLINGS_FILE = """\
import subprocess
from unittest import mock

from widgets import render


def test_x():
    path = "notes.txt"
    assert render.render(["a"]) == "a" and open(path)


def test_mocks():
    with mock.patch("widgets.render.render"):
        pass


def test_reads():
    path = "src/widgets/render.py"
    assert open(path).read()


def test_runs():
    subprocess.run(["true"])
"""


def scoped(source: str, scope: Scope, path: str = "tests/test_repro.py", facts: PatchFacts = FACTS) -> list[str]:
    return [str(v) for v in check_repro_files({path: source}, facts, {path: scope})]


def test_a_scope_reads_the_named_case_and_never_its_siblings() -> None:
    x = (CaseRef((), "test_x"),)
    rules_hit = {v.split(" ", 1)[0] for v in scoped(SIBLINGS_FILE, WHOLE_FILE)}
    assert rules_hit == {"infra", "mocks_changed", "reflection"}
    # Named alone, test_x reads only its own 'path' (notes.txt), never a sibling's binding of the same name.
    assert scoped(SIBLINGS_FILE, Scope(cases=x, own=True)) == []
    # As the repro's own writing, the module-level import of subprocess is read too.
    assert scoped(SIBLINGS_FILE, Scope(cases=x)) == [
        "infra at tests/test_repro.py:1: imports subprocess: the test runs another process"
    ]
    for sibling, rule in (("test_mocks", "mocks_changed"), ("test_reads", "reflection"), ("test_runs", "infra")):
        found = scoped(SIBLINGS_FILE, Scope(cases=(CaseRef((), sibling),), own=True))
        assert [v.split(" ", 1)[0] for v in found] == [rule], (sibling, found)


def test_a_case_the_file_does_not_define_reads_the_whole_file_and_no_case_reads_nothing() -> None:
    assert len(scoped(SIBLINGS_FILE, Scope(cases=(CaseRef((), "test_generated"),), own=True))) == 4
    assert scoped(SIBLINGS_FILE, Scope(cases=(), own=True)) == []


def test_writing_back_a_changed_test_file_is_no_violation_only_as_the_repository_s_own_test_module() -> None:
    facts = PatchFacts(
        files=frozenset({"src/widgets/render.py", "tests/test_render.py", "tests/helpers.py"}),
        names_by_module={"widgets.render": frozenset({"render"})},
        packages=frozenset({"widgets"}),
    )
    clean = "def test_x():\n    pass\n"
    x = (CaseRef((), "test_x"),)
    over = "mocks_changed at {}:1: writes over a file the patch changed"
    assert scoped(clean, Scope(cases=x, own=True), "tests/test_render.py", facts) == []
    assert scoped(clean, Scope(cases=x), "tests/test_render.py", facts) == [over.format("tests/test_render.py")]
    assert scoped(clean, Scope(own=True), "tests/test_render.py", facts) == []  # read whole, still written back
    for not_a_test_module in ("src/widgets/render.py", "tests/helpers.py"):  # base code or a helper: never exempt
        assert scoped(clean, Scope(cases=(), own=True), not_a_test_module, facts) == [over.format(not_a_test_module)]
