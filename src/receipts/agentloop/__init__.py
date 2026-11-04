"""AT1, the attacker loop, and the correctness attacker built on it (AT2's stage 2, at its minimum).

The loop's design is after mini-swe-agent (the SWE-agent team: Yang et al., NeurIPS 2024; MIT license): bash is the
only action, every model answer holds exactly one fenced block, every command runs in a fresh shell, the history is
linear, and the run stops at a step limit or a cost limit. No mini-swe-agent code is copied here and it is not a
dependency; the loop runs over the shared contracts only (ModelClient.complete and Sandbox.run).
"""
