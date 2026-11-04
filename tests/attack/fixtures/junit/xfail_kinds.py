"""The test module two recordings run: xfail_kinds.xml (pytest 9.1.1, by make_samples.py) and pytest3_xfail.json
(pytest 3.3.1 in the django__django-11999 image, by ../runners/record.py). It holds every way a test fails as expected
(an xfail marker, pytest.xfail() at run time, an xfail that never runs, unittest's expectedFailure), a skip whose
reason reads like pytest 3's xfail message, and a pass. It is a fixture, never collected here."""

import unittest

import pytest


@pytest.mark.xfail(reason="known bug")
def test_marked():
    assert 1 + 1 == 3


def test_imperative():
    pytest.xfail("decided while running")


@pytest.mark.xfail(run=False, reason="never run")
def test_not_run():
    raise SystemExit("an xfail with run=False never runs")


@pytest.mark.skip(reason="expected test failure")
def test_skip_whose_reason_reads_like_an_xfail():
    pass


def test_passes():
    assert True


class T(unittest.TestCase):
    @unittest.expectedFailure
    def test_unittest_expected_failure(self):
        self.assertEqual(1, 2)
