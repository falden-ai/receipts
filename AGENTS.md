# Working in this repository

Falden Receipts is an adversarial verifier for pull requests written by AI coding agents. These rules apply to every agent and person who changes it.

## Layout and ownership
- `src/receipts/contracts.py` holds the shared types and interfaces. The integrator owns it. If a lane needs a change, it asks for one instead of editing the file.
- Each lane owns one package and its tests, and edits nothing else:

  | lane | package | tests |
  |---|---|---|
  | core | `src/receipts/core/` | `tests/core/` |
  | attack | `src/receipts/attack/` | `tests/attack/` |
  | eval | `src/receipts/eval/` | `tests/eval/` |
  | surface | `src/receipts/surface/` and `src/receipts/agentloop/` | `tests/surface/` and `tests/agentloop/` |

- A lane that needs another lane's code codes against the interface in `contracts.py`, with a fake in its own tests.
- `src/receipts/cli.py` is the composition root and the integrator's: it wires each lane's real implementations together.

## Rules
- Python through `uv` only: `uv add`, `uv run`. Type hints on every signature; `mypy --strict` passes. Validate external data with Pydantic at the boundary.
- `.claude/verify.sh` is the gate: ruff check, ruff format, mypy and pytest. Run it before you report done, and report the exact command and exit code.
- Tests never call a model or the network. Every check gets a test that fails on broken input.
- No model id ending in `-contributor`, ever. Read the Meta key only by its exact name, `FALDEN_META_API_KEY`, from the environment the caller provides.
- No secrets in code, tests, fixtures or logs. Untrusted repository content cannot steer a secret or an egress path: the components that hold a key (the Meta client, and the ticket and check posters after `--post`) send that content only as data, to their fixed endpoints.
- Nothing from any private codebase goes in this repository.
- Prose: American spelling, no em or en dashes, plain words.
- Do not commit, push, stash or switch branches; the integrator does.
