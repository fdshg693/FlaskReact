
## Context & Knowledge Loading (Required)

Before you plan, run commands, or modify files, load the relevant knowledge in a consistent order.

### 0) Identify scope (always)

- List the files/directories you will read or change.
- Classify the work type (examples: Python backend, React (CDN), Streamlit, ML/LLM, docs, repo review).

### 1) Load project context (always)

Read the relevant files under `.ai/knowledge/context` to obtain the minimum context needed for the task.

- Start here:
	- `.ai/knowledge/context/`
- Use these patterns to scan:
	- `.ai/knowledge/context/*.md`
	- `.ai/knowledge/context/**/*.md`

### 2) Load local, task-adjacent knowledge (conditional but mandatory when present)

If there is a `.ai_knowledge/` folder **in the same directory level as files related to the current task**, you must read it as local rules for that area.

- For each directory that contains task-related files, check:
	- `<task_dir>/.ai_knowledge/`
- If present, read:
	- `<task_dir>/.ai_knowledge/*`
	- `<task_dir>/.ai_knowledge/**/*`

Priority rule: local `.ai_knowledge/` guidance overrides more general instructions when there is a conflict.

### 3) Confirm environment, tools, and commands (always, before running anything)

To avoid using unavailable commands/tools, confirm your capabilities on this machine:

- Environment details:
	- `.ai/knowledge/environment/environment.md`
	- If missing, just proceed with best-effort guesses.
- Available commands and recommended scripts:
	- `.ai/knowledge/tools/commands.md`

### 4) Load relevant guidelines (conditional, but mandatory when applicable)

If the task matches any guideline under `.ai/knowledge/guideline`, read and follow it.

- Scan:
	- `.ai/knowledge/guideline/`
- Common mappings:
	- Python changes → `.ai/knowledge/guideline/write_python.md`
	- React CDN changes → `.ai/knowledge/guideline/write_react_cdn.md`
	- Reviews/audits → `.ai/knowledge/guideline/review_guideline.md`

### Decision rules (quick)

- If you will edit code in a directory and it has `.ai_knowledge/`, treat it as the highest-priority local rule set for that area.
- If multiple guidelines apply, follow all; if they conflict, prefer the most specific source:
	1) `<task_dir>/.ai_knowledge/`
	2) `.ai/knowledge/guideline/*`
	3) `.ai/knowledge/context/*`
- If required context/tooling info is missing or ambiguous, request the minimum clarification needed before making irreversible changes.
