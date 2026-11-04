"""PL1, the planner, against a scripted ModelClient: the model proposes and code decides the plan, its caps, masks,
lenses and temperatures, and every call's cost is kept, the failed ones' included."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from receipts.contracts import (
    Claim,
    ClaimType,
    ContextMask,
    Lens,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    Span,
    SpanSource,
    StageCost,
    Target,
    Tier,
    UncheckedReason,
    Usage,
)
from receipts.core.planner import (
    FALLBACK_MODEL,
    MAX_ATTACKS_PER_LENS,
    MAX_LENSES,
    PLAN_CAPS,
    PLAN_STAGE,
    PLANNER_MODEL,
    PlanCaps,
    PlannedAttack,
    PlanOutput,
    TemperatureMode,
    attack_temperature,
    build_plan,
    plan_attacks,
    run_lenses,
)

WONDER, SPARK = PLANNER_MODEL, FALLBACK_MODEL


def target(tier: Tier = Tier.A, **changes: object) -> Target:
    fields: dict[str, object] = {
        "id": "verified~django__django-11099",
        "tier": tier,
        "repo": "django/django",
        "base_sha": "a" * 40,
        "patch": "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1 +1 @@\n-a\n+b\n",
        "issue": "Widgets crash on an empty list.",
    }
    fields.update(changes)
    return Target.model_validate(fields)


def claim(claim_id: str, claim_type: ClaimType, *, high_risk: bool = False) -> Claim:
    return Claim(
        id=claim_id,
        type=claim_type,
        text=f"{claim_type.value} holds",
        span=Span(source=SpanSource.TEMPLATE, text="quoted text"),
        high_risk=high_risk,
    )


FIXED = claim("fixed_x", ClaimType.FIXED_X)
NOTHING = claim("nothing_else_changed", ClaimType.NOTHING_ELSE_CHANGED)


def proposal(*claims: tuple[str, Sequence[tuple[str, str]]]) -> str:
    return json.dumps(
        {
            "claims": [
                {"claim_id": cid, "attacks": [{"lens": lens, "hypothesis": h} for lens, h in attacks]}
                for cid, attacks in claims
            ]
        }
    )


class Planner:
    """Scripted answers or errors per model; each answer costs 100 micro-dollars on Wonder and 10 on Spark. An answer
    given as (text, finish_reason) finishes that way."""

    def __init__(self, **script: list[str | tuple[str, str] | ModelCallError]) -> None:
        self.script = {model.replace("_", "-"): steps for model, steps in script.items()}
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        step = self.script[request.model].pop(0)
        if isinstance(step, ModelCallError):
            raise step
        text, finish = step if isinstance(step, tuple) else (step, "stop")
        price = 100 if request.model == WONDER else 10
        return ModelResponse(
            model=request.model,
            text=text,
            finish_reason=finish,
            usage=Usage(input_tokens=50, output_tokens=20, reasoning_tokens=5),
            cost=StageCost(stage=request.stage, model=request.model, micro_usd=price, seconds=2.0),
        )


def three(lens: str) -> list[tuple[str, str]]:
    return [(lens, f"{lens} hypothesis {n}") for n in range(1, 4)]


def test_wonder_plans_three_masked_attacks_per_claim_within_the_caps() -> None:
    model = Planner(
        **{WONDER: [proposal(("fixed_x", three("correctness")), ("nothing_else_changed", three("regression")))]}
    )
    result = plan_attacks(target(), [FIXED, NOTHING], model, temperatures=TemperatureMode.SPEC)

    assert (result.model, result.fallback) == (WONDER, False)
    attacks = result.plan.attacks
    assert len(attacks) == 6 and result.skipped == () and result.plan.unchecked == ()
    by_claim = {cid: [a for a in attacks if a.claim_id == cid] for cid in ("fixed_x", "nothing_else_changed")}
    for cid, mine in by_claim.items():
        # Heterogeneous masks: the diff alone, the diff with the touched functions, the claim as a terse spec.
        assert [a.mask for a in mine] == [ContextMask.DIFF, ContextMask.DIFF_FUNCTIONS, ContextMask.CLAIM_SPEC], cid
        assert [a.temperature for a in mine] == [0.0, 0.8, 0.8]  # the first prompt at 0, the rest at 0.8
    assert {a.lens for a in by_claim["fixed_x"]} == {Lens.CORRECTNESS}
    assert [a.id for a in attacks] == [
        "correctness-1",
        "regression-1",
        "correctness-2",
        "regression-2",
        "correctness-3",
        "regression-3",
    ]  # round robin: every claim's first attack before any claim's second
    assert len({a.id for a in attacks}) == 6

    (request,) = model.requests
    assert (request.stage, request.model, request.reasoning_effort, request.temperature) == (
        PLAN_STAGE,
        WONDER,
        "high",
        0.0,
    )
    assert request.json_schema == PlanOutput.model_json_schema()
    assert '<claim id="fixed_x" type="fixed_x" attacks="3">' in request.messages[-1].content
    assert "<diff>" in request.messages[-1].content
    assert "- regression:" in request.messages[0].content and "differential" not in request.messages[0].content
    assert [c.micro_usd for c in result.costs] == [100]


def test_meta_default_runs_every_attack_at_one_and_the_planner_call_stays_at_zero() -> None:
    model = Planner(**{WONDER: [proposal(("fixed_x", three("correctness")))]})
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.META_DEFAULT)
    assert [a.temperature for a in result.plan.attacks] == [1.0, 1.0, 1.0]
    assert model.requests[0].temperature == 0.0
    assert attack_temperature(TemperatureMode.SPEC, 0) == 0.0
    assert attack_temperature(TemperatureMode.SPEC, 4) == 0.8
    assert attack_temperature(TemperatureMode.META_DEFAULT, 0) == 1.0


def test_a_high_risk_claim_gets_five_masks_and_the_caps_list_the_overflow_unchecked() -> None:
    risky = claim("fixed_x", ClaimType.FIXED_X, high_risk=True)
    five = [("correctness", f"c{n}") for n in range(5)]
    model = Planner(**{WONDER: [proposal(("fixed_x", five))]})
    result = plan_attacks(target(), [risky], model, temperatures=TemperatureMode.SPEC)

    kept = result.plan.attacks
    assert [a.mask for a in kept] == [ContextMask.DIFF, ContextMask.DIFF_FUNCTIONS, ContextMask.CLAIM_SPEC]
    assert MAX_ATTACKS_PER_LENS == 3
    assert [(s.attack_id, s.lens, s.reason) for s in result.skipped] == [
        ("correctness-4", Lens.CORRECTNESS, UncheckedReason.BUDGET),
        ("correctness-5", Lens.CORRECTNESS, UncheckedReason.BUDGET),
    ]
    assert "mask diff_tests: over the plan's cap of 3 attacks per lens" in result.skipped[0].detail
    assert [u.what for u in result.plan.unchecked] == ["correctness-4", "correctness-5"]  # listed, never dropped


def test_at_most_four_lenses_are_planned() -> None:
    claims = [claim(f"adds_behavior:{n:010x}", ClaimType.ADDS_BEHAVIOR, high_risk=True) for n in range(2)]
    lenses = ["regression", "correctness", "differential", "crash_restart", "intent_risk"]
    model = Planner(**{WONDER: [proposal((claims[0].id, [(lens, lens) for lens in lenses]), (claims[1].id, []))]})
    result = plan_attacks(target(Tier.LIVE), claims, model, temperatures=TemperatureMode.SPEC)
    assert MAX_LENSES == 4
    assert len({a.lens for a in result.plan.attacks}) == 4
    over = [s for s in result.skipped if "lenses" in s.detail]
    assert [(s.claim_id, s.lens) for s in over] == [(claims[0].id, Lens.INTENT_RISK)]
    # The second claim got no proposals: code fills its five masks with plain attacks on its default lens.
    filled = [a for a in result.plan.attacks if a.claim_id == claims[1].id]
    assert filled and all(a.lens is Lens.CORRECTNESS and "breaks this claim" in a.hypothesis for a in filled)


# ---------------------------------------------------------------- REG-616: the fix's fresh attack plans with no caps

NO_CAPS = PlanCaps(max_lenses=None, max_attacks_per_lens=None)


def test_the_default_plan_caps_are_the_caps_of_every_run() -> None:
    assert PLAN_CAPS == PlanCaps() == PlanCaps(max_lenses=MAX_LENSES, max_attacks_per_lens=MAX_ATTACKS_PER_LENS)
    assert PLAN_CAPS.text() == "at most 4 lenses and at most 3 attacks per lens"
    assert NO_CAPS.text() == "no lens cap and no per-lens cap"


@pytest.mark.parametrize(("lenses", "per_lens"), [(0, 3), (4, 0), (4, -1)])
def test_a_plan_cap_is_at_least_one_or_none(lenses: int, per_lens: int) -> None:
    with pytest.raises(ValueError, match="at least 1, or None for no cap"):
        PlanCaps(max_lenses=lenses, max_attacks_per_lens=per_lens)


def test_a_plan_with_no_caps_keeps_every_attack_round_robin() -> None:
    # Two high-risk claims propose 6 correctness and 4 regression attacks between them. The default keeps 3 per lens
    # and lists the other 4 unchecked (budget); with no caps all 10 are kept, in the same order, AT3 on the first.
    claims = [
        claim("fixed_x", ClaimType.FIXED_X, high_risk=True),
        claim("nothing_else_changed", ClaimType.NOTHING_ELSE_CHANGED, high_risk=True),
    ]
    proposed = proposal(
        ("fixed_x", [("correctness", f"c{n}") for n in range(5)]),
        ("nothing_else_changed", [("correctness", "c5"), *[("regression", f"r{n}") for n in range(4)]]),
    )
    capped = plan_attacks(target(), claims, Planner(**{WONDER: [proposed]}), temperatures=TemperatureMode.SPEC)
    every = plan_attacks(
        target(), claims, Planner(**{WONDER: [proposed]}), temperatures=TemperatureMode.SPEC, caps=NO_CAPS
    )
    kept = ["correctness-1", "correctness-2", "correctness-3", "regression-1", "regression-2", "regression-3"]
    assert [a.id for a in capped.plan.attacks] == kept
    assert [s.attack_id for s in capped.skipped] == ["correctness-4", "correctness-5", "correctness-6", "regression-4"]
    assert not any(note.startswith("planned with") for note in capped.notes)
    assert [a.id for a in every.plan.attacks] == [
        "correctness-1",
        "correctness-2",
        "correctness-3",
        "regression-1",
        "correctness-4",
        "regression-2",
        "correctness-5",
        "regression-3",
        "correctness-6",
        "regression-4",
    ]
    assert every.skipped == () and every.plan.unchecked == ()
    assert [a.id for a in every.plan.attacks if a.existing_tests] == ["regression-1"]
    assert "planned with no lens cap and no per-lens cap, not the default: 10 of 10 attacks kept" in every.notes


def test_a_plan_with_no_caps_keeps_a_fifth_lens() -> None:
    claims = [claim(f"adds_behavior:{n:010x}", ClaimType.ADDS_BEHAVIOR, high_risk=True) for n in range(2)]
    lenses = ["regression", "correctness", "differential", "crash_restart", "intent_risk"]
    model = Planner(**{WONDER: [proposal((claims[0].id, [(lens, lens) for lens in lenses]), (claims[1].id, []))]})
    result = plan_attacks(target(Tier.LIVE), claims, model, temperatures=TemperatureMode.SPEC, caps=NO_CAPS)
    assert len({a.lens for a in result.plan.attacks}) == 5 and len(result.plan.attacks) == 10
    assert result.skipped == ()


def test_tier_a_attacks_only_through_regression_and_correctness() -> None:
    model = Planner(
        **{WONDER: [proposal(("fixed_x", [("differential", "d"), ("mutation", "m"), ("regression", "r")]))]}
    )
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert [a.lens for a in result.plan.attacks] == [Lens.CORRECTNESS, Lens.CORRECTNESS, Lens.REGRESSION]
    assert any("lens differential is not used in this run" in note for note in result.notes)
    assert run_lenses(target()) == (Lens.REGRESSION, Lens.CORRECTNESS)
    assert run_lenses(target(Tier.LIVE), [Lens.CORRECTNESS, Lens.MUTATION]) == (Lens.CORRECTNESS,)


def test_a_run_that_adds_the_differential_lens_plans_it_for_tier_a_and_only_where_it_is_wired() -> None:
    # `receipts run --lenses regression,correctness,differential` (AT4): the default is unchanged.
    wired = [Lens.REGRESSION, Lens.CORRECTNESS, Lens.DIFFERENTIAL]
    added = (Lens.REGRESSION, Lens.CORRECTNESS, Lens.DIFFERENTIAL)
    assert run_lenses(target(), wired) == (Lens.REGRESSION, Lens.CORRECTNESS)
    assert run_lenses(target(), wired, tier_a=added) == added
    assert run_lenses(target(), [Lens.REGRESSION, Lens.CORRECTNESS], tier_a=added) == (
        Lens.REGRESSION,
        Lens.CORRECTNESS,
    )
    model = Planner(
        **{WONDER: [proposal(("fixed_x", [("differential", "d"), ("correctness", "c"), ("regression", "r")]))]}
    )
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC, lenses=wired, tier_a=added)
    assert [a.lens for a in result.plan.attacks] == [Lens.DIFFERENTIAL, Lens.CORRECTNESS, Lens.REGRESSION]
    model = Planner(
        **{WONDER: [proposal(("fixed_x", [("differential", "d"), ("correctness", "c"), ("regression", "r")]))]}
    )
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC, lenses=wired)
    assert Lens.DIFFERENTIAL not in [a.lens for a in result.plan.attacks]


def test_unknown_and_repeated_claims_in_the_proposal_are_ignored() -> None:
    answer = proposal(
        ("fixed_x", three("correctness")), ("fixed_x", three("regression")), ("made_up", three("regression"))
    )
    result = plan_attacks(target(), [FIXED], Planner(**{WONDER: [answer]}), temperatures=TemperatureMode.SPEC)
    assert {a.lens for a in result.plan.attacks} == {Lens.CORRECTNESS}  # the first proposal for fixed_x counts
    assert {a.claim_id for a in result.plan.attacks} == {"fixed_x"}
    assert any("made_up" in note for note in result.notes)


def test_spark_plans_when_wonder_raises_and_both_calls_costs_are_kept() -> None:
    wonder_failed = ModelCallError(
        "cap", UncheckedReason.BUDGET, cost=StageCost(stage=PLAN_STAGE, model=WONDER, micro_usd=40)
    )
    model = Planner(**{WONDER: [wonder_failed], SPARK: [proposal(("fixed_x", three("correctness")))]})
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert (result.model, result.fallback) == (SPARK, True)
    assert len(result.plan.attacks) == 3
    assert [c.micro_usd for c in result.costs] == [40, 10]
    assert [r.model for r in model.requests] == [WONDER, SPARK]
    assert model.requests[1].reasoning_effort == "high"
    # The plan names the fallback: Wonder's failed plan, with its reason, first on its unchecked list.
    (named,) = result.plan.unchecked
    assert (named.what, named.reason) == (f"{PLAN_STAGE}:{WONDER}", UncheckedReason.BUDGET)
    assert result.fallback_reason == named.detail and "R23" in named.detail


def test_when_both_planners_raise_every_claim_is_unchecked_with_the_reason() -> None:
    wonder_failed = ModelCallError("down", UncheckedReason.RUNTIME, cost=StageCost(stage=PLAN_STAGE, micro_usd=3))
    spark_failed = ModelCallError("timed out", UncheckedReason.TIMEOUT, cost=StageCost(stage=PLAN_STAGE, micro_usd=2))
    model = Planner(**{WONDER: [wonder_failed], SPARK: [spark_failed]})
    result = plan_attacks(target(), [FIXED, NOTHING], model, temperatures=TemperatureMode.SPEC)
    assert result.plan.attacks == ()
    assert [(s.claim_id, s.lens, s.reason) for s in result.skipped] == [
        ("fixed_x", Lens.CORRECTNESS, UncheckedReason.TIMEOUT),
        ("nothing_else_changed", Lens.REGRESSION, UncheckedReason.TIMEOUT),
    ]
    # Wonder's failure comes first, then every claim, each with the reason the last call gave.
    assert [u.what for u in result.plan.unchecked] == [f"{PLAN_STAGE}:{WONDER}", "fixed_x", "nothing_else_changed"]
    assert [c.micro_usd for c in result.costs] == [3, 2]
    assert (result.model, result.fallback) == (None, True)


def test_a_wonder_plan_that_fails_validation_twice_gets_exactly_one_spark_call() -> None:
    # REG-570: Wonder answered, but its plan failed schema validation twice, so Spark makes the plan in one call.
    model = Planner(
        **{WONDER: ["not json", '{"claims": "nope"}'], SPARK: [proposal(("fixed_x", three("correctness")))]}
    )
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert [r.model for r in model.requests] == [WONDER, WONDER, SPARK]
    assert (result.model, result.fallback) == (SPARK, True)
    assert len(result.plan.attacks) == 3
    assert [c.micro_usd for c in result.costs] == [100, 100, 10]  # each call's cost, once
    assert [(c.model, c.micro_usd) for c in result.costs] == [(WONDER, 100), (WONDER, 100), (SPARK, 10)]
    (named,) = result.plan.unchecked  # the plan names the fallback
    assert (named.what, named.reason) == (f"{PLAN_STAGE}:{WONDER}", UncheckedReason.UNPARSED)
    assert "failed schema validation twice" in named.detail and "REG-570" in named.detail
    assert result.fallback_reason == named.detail
    assert model.requests[2].messages == model.requests[0].messages  # Spark gets Wonder's first request, not the retry


def test_a_fallback_plan_that_fails_validation_is_not_retried_and_every_cost_is_kept() -> None:
    model = Planner(**{WONDER: ["not json", '{"claims": "nope"}'], SPARK: ["not json either"]})
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert [r.model for r in model.requests] == [WONDER, WONDER, SPARK]  # one Spark call, no retry of its own
    assert result.plan.attacks == ()
    assert [s.reason for s in result.skipped] == [UncheckedReason.UNPARSED]
    assert [u.what for u in result.plan.unchecked] == [f"{PLAN_STAGE}:{WONDER}", "fixed_x"]
    assert [c.micro_usd for c in result.costs] == [100, 100, 10]
    assert (result.model, result.fallback) == (None, True)


@pytest.mark.parametrize("cut", [(proposal(("fixed_x", three("correctness"))), "length"), ("", "stop")])
def test_a_truncated_wonder_plan_gets_exactly_one_spark_call(cut: tuple[str, str]) -> None:
    """REG-577 extends REG-570: a Wonder plan cut at the token limit, or empty, is not retried, and one Spark call
    makes the plan, named on the plan and the result like any fallback."""
    model = Planner(**{WONDER: [cut], SPARK: [proposal(("fixed_x", three("correctness")))]})
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert [r.model for r in model.requests] == [WONDER, SPARK]
    assert (result.model, result.fallback) == (SPARK, True) and len(result.plan.attacks) == 3
    assert [(c.model, c.micro_usd) for c in result.costs] == [(WONDER, 100), (SPARK, 10)]
    (named,) = result.plan.unchecked
    assert (named.what, named.reason) == (f"{PLAN_STAGE}:{WONDER}", UncheckedReason.UNPARSED)
    assert "was truncated" in named.detail and "REG-570" in named.detail and result.fallback_reason == named.detail
    assert model.requests[1].messages == model.requests[0].messages


def test_a_truncated_spark_fallback_is_not_retried_and_leaves_the_claims_unchecked() -> None:
    model = Planner(**{WONDER: [("", "length")], SPARK: [("", "length")]})
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert [r.model for r in model.requests] == [WONDER, SPARK]  # exactly one Spark call
    assert result.plan.attacks == () and (result.model, result.fallback) == (None, True)
    assert [s.reason for s in result.skipped] == [UncheckedReason.UNPARSED]
    assert [u.what for u in result.plan.unchecked] == [f"{PLAN_STAGE}:{WONDER}", "fixed_x"]
    assert [c.micro_usd for c in result.costs] == [100, 10]


class Reasoner:
    """A planner that reasons for `needs` output tokens before its answer, as Wonder does at effort high: under a
    smaller max_tokens it is cut off (finish_reason length, every token spent), as the pilot's were."""

    def __init__(self, needs: int) -> None:
        self.needs = needs
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        cut = request.max_tokens < self.needs
        spent = request.max_tokens if cut else self.needs
        return ModelResponse(
            model=request.model,
            text="" if cut else proposal(("fixed_x", three("correctness"))),
            finish_reason="length" if cut else "stop",
            usage=Usage(input_tokens=2_229, output_tokens=spent, reasoning_tokens=spent - 2),
            cost=StageCost(stage=request.stage, model=request.model, micro_usd=spent * 20, seconds=60.0),
        )


@pytest.mark.parametrize("needs", [12_001, 12_500, 20_000])
def test_a_plan_that_needs_more_than_twelve_thousand_tokens_is_still_made(needs: int) -> None:
    # The 10/28 pilot: Wonder's plans for sympy-14248, django-11728 and django-11964 stopped at exactly 12,000 output
    # tokens, 11,862 to 11,998 of them reasoning, and both arms lost every attack on those targets.
    model = Reasoner(needs)
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert [r.model for r in model.requests] == [WONDER]
    assert len(result.plan.attacks) == 3 and result.skipped == ()
    assert [c.micro_usd for c in result.costs] == [needs * 20]


def test_a_planner_retry_that_raises_keeps_the_first_attempts_cost() -> None:
    retry_failed = ModelCallError(
        "cap", UncheckedReason.BUDGET, cost=StageCost(stage=PLAN_STAGE, model=WONDER, micro_usd=7)
    )
    model = Planner(**{WONDER: ["not json", retry_failed], SPARK: [proposal(("fixed_x", three("correctness")))]})
    result = plan_attacks(target(), [FIXED], model, temperatures=TemperatureMode.SPEC)
    assert result.fallback
    assert [c.micro_usd for c in result.costs] == [107, 10]  # Wonder's two attempts, then Spark's call


def test_no_claims_make_no_call_and_an_empty_plan() -> None:
    model = Planner()
    result = plan_attacks(target(), [], model, temperatures=TemperatureMode.SPEC)
    assert (result.plan.attacks, result.skipped, model.requests) == ((), (), [])


def test_build_plan_is_pure_code_over_any_proposal() -> None:
    plan, skipped, notes = build_plan(
        target(),
        [FIXED],
        {"fixed_x": [PlannedAttack(lens=Lens.REGRESSION, hypothesis="  spaced   out  ")]},
        temperatures=TemperatureMode.SPEC,
        lenses=(Lens.REGRESSION, Lens.CORRECTNESS),
    )
    assert plan.attacks[0].hypothesis == "spaced out"
    assert [a.lens for a in plan.attacks] == [Lens.REGRESSION, Lens.CORRECTNESS, Lens.CORRECTNESS]
    assert skipped == ()
    assert notes == ("fixed_x: 2 of 3 attacks filled in by code",)
    with pytest.raises(ValueError, match="at least one lens"):
        build_plan(target(), [FIXED], {}, temperatures=TemperatureMode.SPEC, lenses=())


# ---------------------------------------------------------------- fix round 5: one AT3 attack per target


def plan_of(proposals: dict[str, list[str]], claims: Sequence[Claim] = (FIXED, NOTHING)) -> list[tuple[str, str, bool]]:
    plan, _, _ = build_plan(
        target(),
        claims,
        {cid: [PlannedAttack(lens=Lens(lens), hypothesis="h") for lens in lenses] for cid, lenses in proposals.items()},
        temperatures=TemperatureMode.SPEC,
        lenses=(Lens.REGRESSION, Lens.CORRECTNESS),
    )
    return [(a.id, a.claim_id, a.existing_tests) for a in plan.attacks]


def test_a_target_gets_one_existing_tests_attack_and_model_attacks_for_the_rest_of_its_regression_slots() -> None:
    # The pilot planned AT3 three times per target, the same deterministic run each time; now it runs once, and the
    # other two regression slots go to the model attacker that writes new tests of unchanged behavior.
    assert plan_of({"fixed_x": ["correctness"] * 3, "nothing_else_changed": ["regression"] * 3}) == [
        ("correctness-1", "fixed_x", False),
        ("regression-1", "nothing_else_changed", True),
        ("correctness-2", "fixed_x", False),
        ("regression-2", "nothing_else_changed", False),
        ("correctness-3", "fixed_x", False),
        ("regression-3", "nothing_else_changed", False),
    ]


def test_at3_goes_to_a_nothing_else_changed_claim_first_and_else_to_the_first_regression_attack() -> None:
    mixed = plan_of({"fixed_x": ["regression", "correctness", "correctness"], "nothing_else_changed": ["regression"]})
    assert [(i, c) for i, c, suite in mixed if suite] == [("regression-2", "nothing_else_changed")]
    alone = plan_of({"fixed_x": ["correctness", "regression", "regression"]}, claims=(FIXED,))
    assert [(i, suite) for i, _, suite in alone] == [
        ("correctness-1", False),
        ("regression-1", True),
        ("regression-2", False),
    ]
    assert not any(suite for _, _, suite in plan_of({"fixed_x": ["correctness"] * 3}, claims=(FIXED,)))


def test_at3_s_attack_is_one_the_caps_kept() -> None:
    # Three claims' first regression slots fill the lens's cap of 3 before the nothing_else_changed claim's first one,
    # so AT3 takes the first kept regression attack rather than one the caps leave out.
    compatible = claim("backward_compatible:0123456789", ClaimType.BACKWARD_COMPATIBLE)
    passing = claim("tests_pass:0123456789", ClaimType.TESTS_PASS)
    claims = (FIXED, passing, compatible, NOTHING)
    plan, skipped, _ = build_plan(
        target(),
        claims,
        {c.id: [PlannedAttack(lens=Lens.REGRESSION, hypothesis="h")] * 3 for c in claims},
        temperatures=TemperatureMode.SPEC,
        lenses=(Lens.REGRESSION, Lens.CORRECTNESS),
    )
    assert [(a.id, a.claim_id, a.existing_tests) for a in plan.attacks] == [
        ("regression-1", "fixed_x", True),
        ("regression-2", passing.id, False),
        ("regression-3", compatible.id, False),
    ]
    assert skipped[0].attack_id == "regression-4" and skipped[0].claim_id == "nothing_else_changed"
