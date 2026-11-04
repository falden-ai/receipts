"""changed_files: the paths a patch leaves in the patched tree, which the run classifier types crashes against."""

from __future__ import annotations

from receipts.agentloop.diff import changed_files

MODIFIED = """diff --git a/src/pkg/core.py b/src/pkg/core.py
index 1111111..2222222 100644
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -1,2 +1,4 @@
 def first(items):
-    return items[0]
+    if not items:
+        return None
+    return items[0]
"""

ADDED = """diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1,2 @@
+def test_x():
+    assert True
"""

DELETED = """diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
"""

RENAMED = """diff --git a/a/before.py b/a/after.py
similarity index 90%
rename from a/before.py
rename to a/after.py
--- a/a/before.py
+++ b/a/after.py
@@ -1 +1 @@
-x = 1
+x = 2
"""

BINARY = """diff --git a/img/logo.png b/img/logo.png
index 3333333..4444444 100644
Binary files a/img/logo.png and b/img/logo.png differ
"""

QUOTED = """diff --git "a/docs/caf\\303\\251 notes.py" "b/docs/caf\\303\\251 notes.py"
--- "a/docs/caf\\303\\251 notes.py"
+++ "b/docs/caf\\303\\251 notes.py"
@@ -1 +1 @@
-a
+b
"""

# An added line whose own text starts with "++ b/" reads "+++ b/" in the diff; hunk counting keeps it content.
TRICKY = """diff --git a/notes.txt b/notes.txt
--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,3 @@
 keep
+++ b/not/a/file.py
 keep too
"""


def test_each_kind_of_change() -> None:
    assert changed_files(MODIFIED) == ("src/pkg/core.py",)
    assert changed_files(ADDED) == ("tests/test_new.py",)
    assert changed_files(DELETED) == ()
    assert changed_files(RENAMED) == ("a/after.py",)
    assert changed_files(BINARY) == ("img/logo.png",)
    assert changed_files(QUOTED) == ("docs/caf\N{LATIN SMALL LETTER E WITH ACUTE} notes.py",)


def test_a_line_inside_a_hunk_is_never_a_header() -> None:
    assert changed_files(TRICKY) == ("notes.txt",)


def test_many_files_keep_their_order_once_each() -> None:
    patch = MODIFIED + DELETED + BINARY + ADDED + MODIFIED
    assert changed_files(patch) == ("src/pkg/core.py", "img/logo.png", "tests/test_new.py")


def test_a_plain_unified_diff_without_git_headers() -> None:
    plain = "--- pkg/mod.py\t2026-10-27 10:00:00\n+++ pkg/mod.py\t2026-10-27 10:01:00\n@@ -1 +1 @@\n-a\n+b\n"
    assert changed_files(plain) == ("pkg/mod.py",)
    assert changed_files("") == ()
