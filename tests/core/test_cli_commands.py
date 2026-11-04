"""The core lane's subcommands, run through the real top-level `receipts` main with the core lane wired in the way the
integrator wires it. None calls a model; the one client here talks to httpx.MockTransport."""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from receipts import cli
from receipts.contracts import ModelRequest, Target, Tier
from receipts.core.cli_commands import register
from receipts.core.meta_client import ARM_COMMAND, MetaClient, SpendMeter, default_spend_path

SPARK = "muse-spark-1.3"


@pytest.fixture
def receipts(monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    monkeypatch.setattr(cli, "LANE_COMMANDS", (register,))

    def run(*argv: str) -> int:
        return cli.main(list(argv))

    return run


def subcommands(parser: argparse.ArgumentParser, prefix: str = "") -> Iterator[tuple[str, argparse.ArgumentParser]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield prefix + name, sub
                yield from subcommands(sub, prefix + name + " ")


def target(tier: Tier) -> Target:
    return Target(
        id="swe-1",
        tier=tier,
        repo="octo/widgets",
        base_sha="a" * 40,
        patch="--- a/w.py\n+++ b/w.py\n",
        issue="Widgets crash when the list is empty.",
    )


def test_every_subcommand_sets_func_to_a_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "LANE_COMMANDS", (register,))
    found = dict(subcommands(cli.build_parser()))
    assert set(found) == {"version", "prices", "spend", "spend arm", "claims", "run"}
    for name, parser in found.items():
        assert callable(parser.get_default("func")), name


def test_prices_prints_the_table_with_its_read_date(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert receipts("prices") == 0
    lines = capsys.readouterr().out.splitlines()
    assert "read 2026-10-27" in lines[0]
    rows = {line.split()[0]: line.split()[1:] for line in lines[2:-1]}
    assert rows == {
        "muse-spark-1.1": ["$1.25", "$0.15", "$4.25"],
        "muse-spark-1.2": ["$1.25", "$0.15", "$4.25"],
        "muse-spark-1.3": ["$1.25", "$0.15", "$4.25"],
        "muse-wonder-1.0": ["$4.00", "$0.20", "$20.00"],
    }


def test_claims_prints_the_tier_a_template_claims(
    receipts: Callable[..., int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "target.json"
    path.write_text(target(Tier.A).model_dump_json())
    assert receipts("claims", str(path)) == 0
    claims = json.loads(capsys.readouterr().out)
    assert [c["type"] for c in claims] == ["fixed_x", "nothing_else_changed"]
    assert claims[0]["span"]["text"] == "Widgets crash when the list is empty."


def test_claims_refuses_a_target_that_is_not_tier_a_or_not_a_target(
    receipts: Callable[..., int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "target.json"
    path.write_text(target(Tier.LIVE).model_dump_json())
    assert receipts("claims", str(path)) == 1
    assert "Tier A" in capsys.readouterr().err

    path.write_text('{"id": "x"}')
    assert receipts("claims", str(path)) == 1
    assert receipts("claims", str(tmp_path / "missing.json")) == 1


def test_spend_refuses_a_spend_file_that_does_not_exist(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert receipts("spend") == 1  # a mistyped path never reads as nothing spent
    err = capsys.readouterr().err
    assert f"no spend file at {default_spend_path()}" in err
    assert ARM_COMMAND in err
    assert not default_spend_path().exists()


def test_spend_arm_needs_the_credit_confirmed_then_arms_the_file_the_clients_use(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert receipts("spend", "arm") == 2
    assert "Confirm the Meta credit is applied" in capsys.readouterr().err
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "150000001") == 2
    assert "past the credit is a code change" in capsys.readouterr().err
    assert not default_spend_path().exists()

    assert receipts("spend", "arm", "--credit-confirmed") == 0
    assert f"armed {default_spend_path()}" in capsys.readouterr().out
    (line,) = default_spend_path().read_text().splitlines()
    record = json.loads(line)
    assert (record["kind"], record["credit_confirmed"], record["cap_micro_usd"]) == ("arm", True, 150_000_000)

    # A default client meters against the same file, so its calls now go through.
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = {
            "id": "chatcmpl-1",
            "model": SPARK,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 0},
        }
        return httpx.Response(200, json=body)

    request = ModelRequest.model_validate(
        {
            "stage": "intake",
            "model": SPARK,
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0,
            "max_tokens": 8,
        }
    )
    stand_in = f"LLM_{secrets.randbelow(10**15):015d}_{secrets.token_urlsafe(16)}"  # never a real key
    with MetaClient(api_key=stand_in, transport=httpx.MockTransport(answer)) as client:
        assert client.complete(request).text == "ok"
    assert len(sent) == 1
    assert SpendMeter(default_spend_path()).spent_micro_usd == 1_250


def test_spend_arm_past_the_credit_needs_above_credit_and_stops_at_the_ceiling(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    # Without --above-credit, a cap past the credit gets the refusal it always got, word for word.
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "425000000") == 2
    assert capsys.readouterr().err == (
        "receipts spend arm: refused. The cap is 1 to 150,000,000 micro-dollars ($150.000000, the Meta credit); "
        "raising it past the credit is a code change.\n"
    )
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "425000001", "--above-credit") == 2
    assert "at most the ceiling, 425,000,000 micro-dollars ($425.000000" in capsys.readouterr().err
    assert receipts("spend", "arm", "--credit-confirmed", "--above-credit") == 2  # a forgotten --cap-micro-usd
    assert "give --cap-micro-usd above 150,000,000" in capsys.readouterr().err
    # A forgotten --credit-confirmed is answered with the whole above-credit command, never the bare one.
    assert receipts("spend", "arm", "--cap-micro-usd", "425000000", "--above-credit") == 2
    assert "then run 'receipts spend arm --credit-confirmed --cap-micro-usd 425000000 --above-credit'." in (
        capsys.readouterr().err
    )
    assert not default_spend_path().exists()


def test_spend_arm_above_the_credit_arms_the_file_the_clients_use_up_to_its_cap(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "425000000", "--above-credit") == 0
    captured = capsys.readouterr()
    assert f"armed {default_spend_path()}" in captured.out
    assert (
        "for $425.000000, $275.000000 of it past the $150.000000 credit: model calls may spend up to the cap"
        in captured.out
    )
    assert captured.err == ""
    (line,) = default_spend_path().read_text().splitlines()
    record = json.loads(line)
    assert (record["kind"], record["cap_micro_usd"], record["above_credit"]) == ("arm", 425_000_000, True)

    assert receipts("spend") == 0
    out = capsys.readouterr().out
    assert (
        "for $425.000000, $275.000000 of it past the $150.000000 credit, the operator having confirmed the Meta credit"
        in out
    )
    assert "spent $0.000000 (0 micro-dollars) of the $425.000000 cap" in out  # the line the final's wrapper reads
    stand_in = f"LLM_{secrets.randbelow(10**15):015d}_{secrets.token_urlsafe(16)}"  # never a real key
    with MetaClient(api_key=stand_in, transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        assert client.meter.limit_micro_usd == 425_000_000  # a default client's meter honors the armed cap


def test_spend_arm_above_the_credit_needs_a_named_spend_file_and_spend_says_why_a_default_one_counts_at_the_credit(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RECEIPTS_SPEND_FILE")
    monkeypatch.chdir(tmp_path)
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "425000000", "--above-credit") == 2
    assert "An arm above the credit needs a spend file named with RECEIPTS_SPEND_FILE" in capsys.readouterr().err
    assert not default_spend_path().exists()

    # An above-credit arm planted in that default file, naming the file's own inode, counts at the credit.
    path = default_spend_path()
    path.parent.mkdir(mode=0o700)
    path.touch(mode=0o600)
    arm = {"v": 2, "at": "2026-10-31T12:00:00Z", "kind": "arm", "credit_confirmed": True, "cap_micro_usd": 425_000_000}
    path.write_text(json.dumps({**arm, "inode": path.stat().st_ino, "above_credit": True}) + "\n")
    assert receipts("spend") == 0
    out = capsys.readouterr().out
    assert "armed at 2026-10-31T12:00:00Z for $425.000000, the operator having confirmed the Meta credit" in out
    assert "this arm counts at the $150.000000 credit" in out
    assert "not one named with RECEIPTS_SPEND_FILE" in out
    assert "spent $0.000000 (0 micro-dollars) of the $150.000000 cap" in out


def test_spend_arm_says_when_it_lowers_the_cap_in_effect_and_when_that_turns_the_kill_switch_on(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "425000000", "--above-credit") == 0
    capsys.readouterr()
    meter = SpendMeter(default_spend_path())
    meter.settle(meter.reserve(200_000_000, stage="s", model=SPARK), 200_000_000, estimated=False)
    # The latest arm counts: a default arm now caps the run at the credit, which $200 has already passed.
    assert receipts("spend", "arm", "--credit-confirmed") == 0
    captured = capsys.readouterr()
    assert "for $150.000000: model calls may spend up to the cap" in captured.out
    assert captured.err == (
        "receipts spend arm: this lowers the cap in effect from $425.000000 to $150.000000\n"
        "receipts spend arm: $200.000000 is already spent, so the kill switch is on: every model call is refused\n"
    )
    assert SpendMeter(default_spend_path()).limit_micro_usd == 150_000_000


def test_spend_reports_armed_spent_held_and_the_kill_switch(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert receipts("spend", "arm", "--credit-confirmed", "--cap-micro-usd", "3000") == 0
    capsys.readouterr()
    meter = SpendMeter(default_spend_path())
    meter.settle(meter.reserve(1_000, stage="intake", model=SPARK), 1_000, estimated=False)
    in_flight = meter.reserve(500, stage="intake", model=SPARK)

    assert receipts("spend") == 0
    out = capsys.readouterr().out
    assert f"spend file: {default_spend_path()}" in out
    assert "armed at " in out and "for $0.003000" in out
    assert "spent $0.001000 (1,000 micro-dollars) of the $0.003000 cap" in out
    assert "held $0.000500 at the worst case" in out
    assert "left $0.001500" in out
    assert "kill switch" not in out

    meter.settle(in_flight, 2_500, estimated=False)
    assert receipts("spend") == 0
    assert "the kill switch is on" in capsys.readouterr().out


def test_spend_reports_an_unarmed_file_and_refuses_a_broken_one(
    receipts: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    path = default_spend_path()
    path.parent.mkdir(mode=0o700, parents=True)  # private, as 'spend arm' makes it, so only the content is tested
    path.touch(mode=0o600)
    hold = {"v": 2, "at": "2026-10-27T12:00:00Z", "kind": "hold", "id": "f" * 32, "stage": "s", "model": SPARK}
    settle = {"v": 2, "at": "2026-10-27T12:00:00Z", "kind": "settle", "hold": "f" * 32, "stage": "s", "model": SPARK}
    path.write_text(
        json.dumps({**hold, "micro_usd": 10}) + "\n" + json.dumps({**settle, "micro_usd": 7, "estimated": False}) + "\n"
    )
    assert receipts("spend") == 0
    out = capsys.readouterr().out
    assert "not armed: every model call is refused" in out
    assert "spent $0.000007 (7 micro-dollars); once armed, the cap is at most $150.000000" in out

    path.write_text("not json\n")
    assert receipts("spend") == 1
    assert "invalid line" in capsys.readouterr().err

    path.write_text("")
    path.chmod(0o644)  # a file 'spend arm' did not make, such as one a checkout shipped
    assert receipts("spend") == 1
    assert "has mode 0644, not 0600" in capsys.readouterr().err
