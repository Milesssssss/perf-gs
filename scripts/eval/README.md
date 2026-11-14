# ESC — Expected Safe Changeability harness

Scores a codebase by **how well it absorbs unseen change**, not by how clean it
looks. The headline number in `[0, 100]` is a weighted geometric mean of six
behavioural subscores:

```
ESC = 100 * C^0.40 * E^0.15 * U^0.15 * L^0.15 * T^0.10 * S^0.05
```

| dim | dimension | how it is measured |
|-----|-----------|--------------------|
| C | correctness on future-change tasks | fraction of agent runs that pass hidden tests + gates |
| E | edit efficiency | `exp(-median cost)` normalised vs a baseline |
| U | understandability | graded comprehension Q&A (answer, cited files, calibration, context) |
| L | change locality | concept entropy `1 - H_c` blended with measured patch spread |
| T | testability & regression safety | mutation/fault detection × determinism × hygiene |
| S | static structural quality | AST complexity, size, fan-out, duplication, import cycles |

A geometric mean means one catastrophic dimension drags the whole score down —
"beautifully formatted but impossible to modify" scores poorly. **Hard gates**
(tests, numerical parity, performance, ABI, determinism, mutation floor) run
*before* scoring, so cleanliness can never be traded for correctness.

## Running

```powershell
# agent-free: S + static locality + concept-entropy dashboard (runnable today)
python scripts\eval\run_esc.py analyze

# aggregate six subscores into ESC
python scripts\eval\run_esc.py score -c 0.72 -e 0.61 -u 0.80 -l 0.15 -t 0.55 -s 0.88

# full evaluation with a real agent (see "Wiring an agent")
python scripts\eval\run_esc.py evaluate --family densification --seeds 0,1,2 `
    --agent-cmd "my-agent --prompt-file {prompt_file}" --mutation-json mutation.json

# champion/candidate iteration loop
python scripts\eval\run_esc.py loop --iterations 5 `
    --agent-cmd "eval-agent ..." --refactor-cmd "refactor-agent ..."
```

Self-tests: `.venv\Scripts\python.exe -m pytest scripts\eval\tests -q`

## Task families

The suite lives in [esc/tasks/registry.py](esc/tasks/registry.py). The
`densification` family asks the agent to add a densification mode
([esc/tasks/adjacent.py](esc/tasks/adjacent.py)).

## Wiring an agent

`evaluate` / `loop` run `--agent-cmd` **inside each sandbox**. Command tokens are
`str.format`-expanded with `workspace`, `prompt_file`, `seed`, `task`; the prompt
is also on `$ESC_PROMPT`. The agent edits files in place; the harness reads the
`git diff` for the patch footprint. Optionally the agent drops JSON files in the
workspace:

- `.esc_telemetry.json` → `{tokens, wall_time_s, retries, tool_calls, patch_lines, files_inspected, failed_test_cycles}`
- `.esc_answer.json` (comprehension) → `{answer, cited_files, confidence, context_tokens}`

Use a **different** agent for `--agent-cmd` (evaluation) and `--refactor-cmd`
(refactoring) so the refactorer never grades itself.

## Layout

```
esc/config.py        weights, thresholds, concept definitions
esc/model.py         immutable result/telemetry value objects
esc/scoring/         the six subscore reducers + geometric-mean aggregation
esc/gates.py         hard behavioural gates
esc/agents/          agent boundary (scripted / subprocess)
esc/tasks/           hidden change-task + comprehension-task families
esc/workspace.py     ephemeral sandboxes
esc/evaluate.py      paired evaluation over tasks × seeds
esc/loop.py          champion/candidate iteration + paired bootstrap
esc/stats.py         paired-bootstrap significance
esc/report.py        diagnostic dashboard
esc/cli.py           command-line entry
```

Add a change family by declaring a `CommandChangeTask` (prompt + hidden tests +
classified verification checks) and registering it in
[esc/tasks/registry.py](esc/tasks/registry.py).
