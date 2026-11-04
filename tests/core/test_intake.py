"""IN1, intake. Tier A template claims need no model; model extraction runs against a fake ModelClient. A claim whose
quote is not verbatim in its source is fed in on purpose and must be dropped and reported, never kept."""

from __future__ import annotations

import json
import re

import pytest

from receipts.contracts import (
    ClaimType,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    SpanSource,
    StageCost,
    Target,
    Tier,
    UncheckedReason,
    Usage,
)
from receipts.core.intake import (
    INTAKE_STAGE,
    MAX_SOURCE_CHARS,
    MIN_QUOTE_WORDS,
    TEMPLATE_NOTHING_ELSE_CHANGED,
    DropReason,
    ExtractionOutput,
    count_words,
    extract_claims,
    find_quote,
    intake,
    patch_risk,
    template_claims,
)
from receipts.core.meta_client import SpendNotArmedError

TITLE = "Fix crash on empty widget list"
BODY = "This fixes the crash when the list is empty.\n\nNo other behavior changes. All tests pass."


def tier_a(**changes: object) -> Target:
    fields: dict[str, object] = {
        "id": "swe-1",
        "tier": Tier.A,
        "repo": "octo/widgets",
        "base_sha": "a" * 40,
        "patch": "--- a/w.py\n+++ b/w.py\n",
        "issue": "Widgets crash when the list is empty.\n\nSteps: call render([]).",
    }
    fields.update(changes)
    return Target.model_validate(fields)


def live(**changes: object) -> Target:
    fields: dict[str, object] = {
        "id": "pr-7",
        "tier": Tier.LIVE,
        "repo": "octo/widgets",
        "base_sha": "b" * 40,
        "patch": "--- a/w.py\n+++ b/w.py\n",
        "title": TITLE,
        "body": BODY,
    }
    fields.update(changes)
    return Target.model_validate(fields)


def claim(claim_type: str, source: str, quote: str, text: str = "restated") -> dict[str, str]:
    return {"type": claim_type, "source": source, "quote": quote, "text": text}


class FakeModel:
    """A ModelClient that returns scripted texts and records each request. It never calls anything."""

    def __init__(self, *texts: str, finish: str = "stop") -> None:
        self.texts = list(texts)
        self.finish = finish
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.texts:
            raise AssertionError("the fake model was called more often than scripted")
        return ModelResponse(
            model=request.model,
            text=self.texts.pop(0),
            finish_reason=self.finish,
            usage=Usage(input_tokens=10, output_tokens=5),
        )


def answer(*claims: dict[str, str]) -> str:
    return json.dumps({"claims": list(claims)})


# ---------------------------------------------------------------- Tier A templates


def test_tier_a_gets_the_two_template_claims_with_no_model() -> None:
    target = tier_a()
    fixed, nothing_else = template_claims(target)

    assert fixed.id == "fixed_x"  # relative to the target (contract v2); the target id goes in the finding id
    assert fixed.type is ClaimType.FIXED_X
    assert fixed.span.source is SpanSource.ISSUE
    assert fixed.span.text == target.issue
    assert (fixed.span.start, fixed.span.end) == (0, len(target.issue))

    assert nothing_else.id == "nothing_else_changed"
    assert nothing_else.type is ClaimType.NOTHING_ELSE_CHANGED
    assert nothing_else.span.source is SpanSource.TEMPLATE
    assert nothing_else.span.text == TEMPLATE_NOTHING_ELSE_CHANGED
    assert not fixed.high_risk and not nothing_else.high_risk  # a one-file patch with no risky path

    # The same target always gives the same claims, so every baseline is attacked on the same two.
    assert template_claims(tier_a()) == (fixed, nothing_else)


def test_template_claims_refuse_other_tiers_and_an_empty_issue() -> None:
    for tier in (Tier.B, Tier.C, Tier.LIVE):
        with pytest.raises(ValueError, match="Tier A"):
            template_claims(tier_a(tier=tier))
    with pytest.raises(ValueError, match="issue text"):
        template_claims(tier_a(issue="  \n"))


# ---------------------------------------------------------------- model extraction


def test_extraction_keeps_claims_whose_quotes_are_verbatim_with_their_offsets() -> None:
    target = live()
    model = FakeModel(
        answer(
            claim("fixed_x", "pr_title", "Fix crash on empty widget list"),
            claim("fixed_x", "pr_body", "This fixes the crash when the list is empty."),
            claim("nothing_else_changed", "pr_body", "No other behavior changes."),
            claim("tests_pass", "pr_body", "All tests pass."),
        )
    )
    result = extract_claims(target, model)

    assert [c.type for c in result.claims] == [
        ClaimType.FIXED_X,
        ClaimType.FIXED_X,
        ClaimType.NOTHING_ELSE_CHANGED,
        ClaimType.TESTS_PASS,
    ]
    texts = {SpanSource.PR_TITLE: target.title, SpanSource.PR_BODY: target.body}
    for kept in result.claims:
        assert kept.span.start is not None and kept.span.end is not None
        assert texts[kept.span.source][kept.span.start : kept.span.end] == kept.span.text
        assert re.fullmatch(rf"{kept.type.value}:[0-9a-f]{{10}}", kept.id), kept.id
    assert len({kept.id for kept in result.claims}) == 4
    assert result.dropped == ()
    assert result.unchecked == ()

    (request,) = model.requests
    assert request.stage == INTAKE_STAGE
    assert request.model == "muse-spark-1.3"
    assert request.reasoning_effort == "low"
    assert request.json_schema == ExtractionOutput.model_json_schema()
    prompt = request.messages[-1].content
    assert '<source name="pr_title">' in prompt and TITLE in prompt
    assert '<source name="pr_body">' in prompt and BODY in prompt
    # The same answer gives the same ids, so a rerun dedupes against the first run.
    again = extract_claims(target, FakeModel(answer(claim("tests_pass", "pr_body", "All tests pass."))))
    assert again.claims[0].id == result.claims[3].id


def test_a_claim_whose_quote_is_not_verbatim_is_dropped_and_reported_never_kept() -> None:
    paraphrase = "This fixes the crash for empty lists."
    respaced = "all  tests pass."
    model = FakeModel(
        answer(
            claim("fixed_x", "pr_body", paraphrase),
            claim("tests_pass", "pr_body", respaced),
            claim("perf", "pr_body", " "),  # a single space is in the body, so only the blank guard drops it
            claim("nothing_else_changed", "pr_body", "No other behavior changes."),
        )
    )
    result = extract_claims(live(), model)

    assert [c.type for c in result.claims] == [ClaimType.NOTHING_ELSE_CHANGED]
    assert [(d.type, d.reason) for d in result.dropped] == [
        (ClaimType.FIXED_X, DropReason.QUOTE_NOT_FOUND),
        (ClaimType.TESTS_PASS, DropReason.QUOTE_NOT_FOUND),
        (ClaimType.PERF, DropReason.QUOTE_NOT_FOUND),
    ]
    assert result.dropped[0].detail == "not found in pr_body"
    assert "only once spacing and case are normalized" in result.dropped[1].detail
    assert result.dropped[2].detail == "the quote is blank"
    kept_quotes = {kept.span.text for kept in result.claims}
    assert paraphrase not in kept_quotes and respaced not in kept_quotes


def test_a_claim_naming_a_source_that_was_not_given_is_dropped() -> None:
    # The quote is in the title, but the claim says it came from a session summary nobody passed.
    model = FakeModel(answer(claim("fixed_x", "session_summary", "Fix crash on empty widget list")))
    result = extract_claims(live(), model)
    assert result.claims == ()
    (dropped,) = result.dropped
    assert (dropped.reason, dropped.source) == (DropReason.SOURCE_NOT_GIVEN, SpanSource.SESSION_SUMMARY)


def test_the_targets_commit_messages_and_session_summary_are_sources_checked_like_the_rest() -> None:
    commits = "Handle the empty list in render()\n\nKeeps the old output for every other input."
    summary = "I ran the full test suite and every test passed."
    model = FakeModel(
        answer(
            claim("backward_compatible", "commit_message", "Keeps the old output for every other input."),
            claim("tests_pass", "session_summary", "every test passed"),
            claim("perf", "session_summary", "it is twice as fast"),
        )
    )
    result = extract_claims(live(commit_messages=commits, session_summary=summary), model)

    assert [(c.type, c.span.source) for c in result.claims] == [
        (ClaimType.BACKWARD_COMPATIBLE, SpanSource.COMMIT_MESSAGE),
        (ClaimType.TESTS_PASS, SpanSource.SESSION_SUMMARY),
    ]
    texts = {SpanSource.COMMIT_MESSAGE: commits, SpanSource.SESSION_SUMMARY: summary}
    for kept in result.claims:
        assert kept.span.start is not None and kept.span.end is not None
        assert texts[kept.span.source][kept.span.start : kept.span.end] == kept.span.text
    (dropped,) = result.dropped
    assert (dropped.type, dropped.reason) == (ClaimType.PERF, DropReason.QUOTE_NOT_FOUND)
    prompt = model.requests[0].messages[-1].content
    assert f'<source name="commit_message">\n{commits}\n</source>' in prompt
    assert f'<source name="session_summary">\n{summary}\n</source>' in prompt

    # Without them, a claim citing either is dropped: the model was never shown that source.
    bare = extract_claims(live(), FakeModel(answer(claim("tests_pass", "session_summary", "every test passed"))))
    assert bare.claims == ()
    assert bare.dropped[0].reason is DropReason.SOURCE_NOT_GIVEN


def test_a_repeated_claim_is_dropped_as_a_duplicate() -> None:
    quote = "All tests pass."
    model = FakeModel(answer(claim("tests_pass", "pr_body", quote), claim("tests_pass", "pr_body", quote)))
    result = extract_claims(live(), model)
    assert len(result.claims) == 1
    (dropped,) = result.dropped
    assert dropped.reason is DropReason.DUPLICATE


def test_an_unparsed_extraction_is_reported_unchecked_with_no_claims() -> None:
    model = FakeModel("not json", '{"claims": [{"type": "nope"}]}')
    result = extract_claims(live(), model)
    assert result.claims == ()
    (unchecked,) = result.unchecked
    assert (unchecked.what, unchecked.reason) == ("intake", UncheckedReason.UNPARSED)
    assert len(model.requests) == 2
    assert len(result.responses) == 2


def test_a_truncated_extraction_is_unchecked_and_not_retried() -> None:
    model = FakeModel(answer(claim("tests_pass", "pr_body", "All tests pass.")), finish="length")
    result = extract_claims(live(), model)
    assert result.claims == ()
    (unchecked,) = result.unchecked
    assert unchecked.reason is UncheckedReason.UNPARSED
    assert "truncated" in unchecked.detail
    assert len(model.requests) == 1


@pytest.mark.parametrize(
    "error",
    [
        ModelCallError("timed out", UncheckedReason.TIMEOUT),  # any client's error, through the shared contract
        ModelCallError("runtime failure", UncheckedReason.RUNTIME),
        SpendNotArmedError("the spend meter is not armed"),  # the Meta client's own, a ModelCallError too
    ],
    ids=["timeout", "runtime", "meta-not-armed"],
)
def test_a_failed_call_is_reported_unchecked_with_its_reason(error: ModelCallError) -> None:
    class Broke:
        def complete(self, request: ModelRequest) -> ModelResponse:
            raise error

    result = extract_claims(live(), Broke())
    assert result.claims == ()
    (unchecked,) = result.unchecked
    assert (unchecked.what, unchecked.reason, unchecked.detail) == ("intake", error.reason, str(error))


def test_a_long_source_is_cut_for_the_model_and_reported() -> None:
    body = "No other behavior changes. " + "x" * MAX_SOURCE_CHARS
    model = FakeModel(answer(claim("nothing_else_changed", "pr_body", "No other behavior changes.")))
    result = extract_claims(live(body=body), model)
    assert len(result.claims) == 1
    (unchecked,) = result.unchecked
    assert (unchecked.what, unchecked.reason) == ("intake:pr_body", UncheckedReason.BUDGET)
    prompt = model.requests[0].messages[-1].content
    assert body not in prompt and body[:MAX_SOURCE_CHARS] in prompt


def test_a_target_with_no_agent_text_makes_no_call_and_is_unchecked_for_no_input() -> None:
    for run in (extract_claims, intake):
        model = FakeModel()
        result = run(live(title="", body="  "), model)
        assert result.claims == ()
        assert model.requests == []
        (unchecked,) = result.unchecked
        assert (unchecked.what, unchecked.reason) == ("intake", UncheckedReason.NO_INPUT)


# ---------------------------------------------------------------- round 2: one entry point, the quote rule, risk, costs


def test_intake_gives_tier_a_its_templates_with_no_call_and_other_tiers_extraction() -> None:
    model = FakeModel()
    result = intake(tier_a(), model)
    assert [c.id for c in result.claims] == ["fixed_x", "nothing_else_changed"]
    assert model.requests == []
    assert (result.unchecked, result.costs) == ((), ())

    (unchecked,) = intake(tier_a(issue=" \n"), model).unchecked  # nothing to take fixed_x from
    assert (unchecked.what, unchecked.reason) == ("intake", UncheckedReason.NO_INPUT)
    assert model.requests == []

    extracting = FakeModel(answer(claim("tests_pass", "pr_body", "All tests pass.")))
    assert [c.type for c in intake(live(), extracting).claims] == [ClaimType.TESTS_PASS]
    assert len(extracting.requests) == 1


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        ("the", DropReason.QUOTE_TOO_SHORT),  # a common word is in every body
        ("All tests", DropReason.QUOTE_TOO_SHORT),  # 2 words
        ("ixes the crash", DropReason.QUOTE_SPLITS_A_WORD),  # starts inside "fixes"
        ("fixes the cras", DropReason.QUOTE_SPLITS_A_WORD),  # ends inside "crash"
        ("This fixes the crash when the list is empty.", None),
        ("No other behavior changes.", None),
    ],
)
def test_a_quote_is_at_least_three_words_on_word_boundaries(quote: str, reason: DropReason | None) -> None:
    model = FakeModel(answer(claim("fixed_x", "pr_body", quote)))
    result = extract_claims(live(), model)
    if reason is None:
        (kept,) = result.claims
        assert BODY[kept.span.start : kept.span.end] == quote
    else:
        assert result.claims == ()
        assert [d.reason for d in result.dropped] == [reason]
    assert MIN_QUOTE_WORDS == 3


def test_a_quote_is_found_where_it_stands_on_word_boundaries() -> None:
    text = "Prefixes the cache key. It fixes the cache key bug."
    assert find_quote("fixes the cache", text) == text.index("It fixes") + 3  # not inside "Prefixes"
    assert find_quote("fixes the cache", "Prefixes the cache key.") == (
        DropReason.QUOTE_SPLITS_A_WORD,
        "in {source} only inside a longer word, not on word boundaries",
    )
    assert find_quote("(no behavior change)", "This is (no behavior change) at all.") == len("This is ")


TEST_EDIT = """\
diff --git a/tests/test_widgets.py b/tests/test_widgets.py
--- a/tests/test_widgets.py
+++ b/tests/test_widgets.py
@@ -10,3 +10,3 @@ def test_render():
-    assert render([]) == "empty"
+    assert render([]) == ""
"""
SECURITY_PATH = """\
diff --git a/src/widgets/auth/tokens.py b/src/widgets/auth/tokens.py
--- a/src/widgets/auth/tokens.py
+++ b/src/widgets/auth/tokens.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
"""
ADDED_TEST = """\
diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1,2 @@
+def test_new():
+    assert True
"""


def large_patch(files: int, lines: int) -> str:
    parts = []
    for n in range(files):
        body = "".join(f"+line {i}\n" for i in range(lines))
        parts.append(f"diff --git a/m{n}.py b/m{n}.py\n--- a/m{n}.py\n+++ b/m{n}.py\n@@ -0,0 +1,{lines} @@\n{body}")
    return "".join(parts)


def test_claims_are_high_risk_when_the_patch_edits_tests_touches_security_code_or_is_large() -> None:
    assert patch_risk(TEST_EDIT) == ("edits or removes lines of existing tests: tests/test_widgets.py",)
    assert patch_risk(SECURITY_PATH) == ("touches security-sensitive code: src/widgets/auth/tokens.py",)
    assert patch_risk(large_patch(6, 1)) == ("a large change: 6 files, 6 changed lines",)
    assert patch_risk(large_patch(1, 401)) == ("a large change: 1 files, 401 changed lines",)
    assert patch_risk(large_patch(5, 80)) == ()  # 5 files and 400 lines are not large
    assert patch_risk(ADDED_TEST) == ()  # adding a test edits no existing one
    assert patch_risk("--- a/w.py\n+++ b/w.py\n") == ()

    for risky in (TEST_EDIT, SECURITY_PATH):
        assert all(c.high_risk for c in template_claims(tier_a(patch=risky)))
        model = FakeModel(answer(claim("tests_pass", "pr_body", "All tests pass.")))
        assert all(c.high_risk for c in extract_claims(live(patch=risky), model).claims)
    model = FakeModel(answer(claim("tests_pass", "pr_body", "All tests pass.")))
    assert not any(c.high_risk for c in extract_claims(live(), model).claims)


class Costed:
    """A ModelClient whose answers each cost 11 micro-dollars, and that raises when its script says so."""

    def __init__(self, *steps: str | ModelCallError) -> None:
        self.steps = list(steps)

    def complete(self, request: ModelRequest) -> ModelResponse:
        step = self.steps.pop(0)
        if isinstance(step, ModelCallError):
            raise step
        return ModelResponse(
            model=request.model,
            text=step,
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=StageCost(stage=request.stage, model=request.model, micro_usd=11, seconds=1.0),
        )


def test_intake_records_every_calls_cost_even_when_the_retry_raises() -> None:
    # The first answer fails validation and the retry raises: both calls' costs reach the extraction.
    error = ModelCallError("cap reached", UncheckedReason.BUDGET, cost=StageCost(stage="intake", micro_usd=4))
    result = extract_claims(live(), Costed("not json", error))
    (unchecked,) = result.unchecked
    assert unchecked.reason is UncheckedReason.BUDGET
    (cost,) = result.costs
    assert (cost.stage, cost.micro_usd, cost.seconds) == ("intake", 15, 1.0)

    parsed = extract_claims(live(), Costed(answer(claim("tests_pass", "pr_body", "All tests pass."))))
    assert [c.micro_usd for c in parsed.costs] == [11]
    unparsed = extract_claims(live(), Costed("not json", "still not json"))
    assert [c.micro_usd for c in unparsed.costs] == [11, 11]


# ---------------------------------------------------------------- round 3: what a word is, and what extraction reads


@pytest.mark.parametrize(
    ("quote", "words"),
    [
        ("e.g. a", 2),  # \w+ would count e, g and a
        ("v1.2.3", 1),  # \w+ would count v1, 2 and 3
        ("- - - -", 0),  # punctuation alone is no word
        ("___ a b", 2),  # an underscore is no letter or digit, though \w matches it (round 3 review, N28)
        ("__init__ is fixed", 3),  # a token with letters among underscores is a word
        ("All tests pass.", 3),
        ("(no behavior change)", 3),
        ("i.e. it works", 3),
    ],
)
def test_a_word_is_a_whitespace_separated_token_with_a_letter_or_a_digit(quote: str, words: int) -> None:
    assert count_words(quote) == words
    text = f"Some context: {quote} and more."
    found = find_quote(quote, text)
    if words < MIN_QUOTE_WORDS:
        assert isinstance(found, tuple) and found[0] is DropReason.QUOTE_TOO_SHORT
    else:
        assert found == text.index(quote)


def test_the_issue_text_never_reaches_extraction() -> None:
    # The issue states the problem; the agent's claims are in its own words, so the issue is never a source.
    issue = "ISSUE ONLY SENTENCE: widgets crash when the list is empty."
    model = FakeModel(answer(claim("tests_pass", "pr_body", "All tests pass.")))
    extract_claims(live(issue=issue), model)
    (request,) = model.requests
    assert all("ISSUE ONLY SENTENCE" not in message.content for message in request.messages)
    assert '<source name="issue">' not in request.messages[-1].content

    # A target whose only text is its issue has nothing of the agent's to take claims from, so no call is made.
    silent = FakeModel()
    result = intake(live(title="", body="", issue=issue), silent)
    assert silent.requests == []
    assert [u.reason for u in result.unchecked] == [UncheckedReason.NO_INPUT]
