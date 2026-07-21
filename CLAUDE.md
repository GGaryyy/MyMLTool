# CLAUDE.md — Project Rules

## Workflow
1. Plan first. For any non-trivial task, enter plan mode before writing code. Write the plan to `docs/plan_<topic>.md` — include context, file list, steps, error handling, and how to verify. Ask clarifying questions before finalising; do not assume intent. When modifying an existing feature, update its plan file first, then implement.
2. Before a plan is approved, "ok / go ahead / do it" means proceed to planning. After the plan is approved, it means implement the approved plan.
3. Implement → test → reports → changelog → done. Test tiers and report templates: see the `testing-and-reports` skill. Documentation rules: see the `project-docs` skill.
4. Read existing code before modifying it. Do not add features, refactors, or improvements beyond what was asked. Prefer editing existing files; required project docs are the exception.
5. Validate that changes work before marking a task complete.

## Model Convention
- Plan phase: Fable. Execution: Opus. The user switches the main model manually with `/model`; Claude does not switch it itself.
- Subagents run per role, set in each agent's `model` frontmatter (`.claude/agents/`): explorer → haiku, implementer → sonnet, reviewer → opus. No blanket subagent model override — each agent picks the cheapest model that fits its task.

## Subagents
- Predefined roles live in `.claude/agents/`: `explorer` (read-only search), `implementer` (code against an approved plan), `reviewer` (quality + security review on a stronger model). Prefer these over ad-hoc agents.
- Use parallel subagents when two or more subtasks are genuinely independent: separate modules, multi-area codebase research, or running different test categories. Do not parallelise sequentially dependent work.
- Launch independent agents in a single message. Each agent prompt must be self-contained (file paths, context, clear instructions) — agents have no memory of this conversation.
- After agents complete, synthesise results and resolve conflicts across modules. If an agent fails, diagnose and re-spawn with corrected instructions — do not retry blindly.

## Project Structure
- `src/` — source code, subdirectories by feature or layer. `tests/` — mirrors `src/`; no test files inside `src/`. `docs/` — plans, reports, project docs. `output/` — program-generated files (logs, exports, temp); never in the project root.
- Root files only: `README.md`, entry points (`main.py`, `index.ts`, `app.py`, ...), and config (`package.json`, `pyproject.toml`, `.env`, `Dockerfile`, `docker-compose.yaml`, ...).

## Docker
- Package the project as a Docker container by default, started via `docker compose up` — provide a `docker-compose.yaml`, not just a bare `docker run` command.
- Exclude `Dockerfile` and `docker-compose.yaml` from the image via `.dockerignore`; keep both at the project root.

## Code Style
- Small, single-purpose functions. Readable over clever; three similar lines beat a premature abstraction.
- Constants at the top of the file in a clearly marked block. Never silence exceptions with a bare `except` / `catch` — be specific.
- Comments only where the logic is not self-evident. Do not add docstrings, comments, or type annotations to code you did not change.

## Git
- Never commit or push without explicit user instruction (also enforced via `ask` rules in `.claude/settings.json`). No amend or force-push unless explicitly asked.
- Commit messages explain the "why", not just the "what".

## Environment
- Windows shell is PowerShell: use `$env:VAR = "value"` (`set VAR=value` does not work for Python/Node to read). macOS/Linux: `export VAR="value"`.
- Use forward slashes in code paths; platform-specific slashes only in shell-specific docs.

## Communication
- Reply in the same language the user writes in. Be concise.
- Use tables and code blocks for structured information. Reference code locations as `file_path:line_number`.
