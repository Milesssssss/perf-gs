"""Command-line entry point for the ESC harness.

Subcommands:

    analyze    agent-free: compute S and the static locality term over a codebase
               and print the concept-entropy dashboard (runnable today)
    score      combine six subscores into ESC (the aggregation, in isolation)
    evaluate   full evaluation with an agent (scripted no-op by default; wire a
               real agent with --agent-cmd)
    loop       run the champion/candidate procedure

Only ``analyze`` and ``score`` are useful without a live agent; ``evaluate`` and
``loop`` are the seams for a real coding agent via ``--agent-cmd`` (a shell
command run inside each sandbox).
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from esc.agents.scripted import ScriptedAgentRunner
from esc.agents.subprocess_runner import SubprocessAgentRunner, SubprocessConfig
from esc.config import EscConfig
from esc.evaluate import analyze_static, evaluate_codebase
from esc.loop import ChampionCandidateLoop
from esc.model import MutationResult, SubScores
from esc.report import (
    render_dashboard,
    render_loop,
    render_static_analysis,
    result_to_json,
)
from esc.scoring.aggregate import compute_esc_partial
from esc.tasks.registry import default_families, select_families

_REPO_ROOT = Path(__file__).resolve().parents[3]


# --- Argument parsing -------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esc", description=__doc__)
    parser.add_argument("--root", type=Path, default=_REPO_ROOT,
                        help="codebase root (default: repo root)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="agent-free S + static locality")
    p_analyze.add_argument("--json", action="store_true", help="emit JSON")

    p_score = sub.add_parser("score", help="aggregate six subscores into ESC")
    for dim in ("c", "e", "u", "l", "t", "s"):
        p_score.add_argument(f"-{dim}", type=float, default=None,
                             help=f"{dim.upper()} subscore in [0,1]")
    p_score.add_argument("--subscores", type=Path, help="JSON file of subscores")

    p_eval = sub.add_parser("evaluate", help="full evaluation with an agent")
    _add_agent_args(p_eval)
    p_eval.add_argument("--json", action="store_true")

    p_loop = sub.add_parser("loop", help="champion/candidate iteration")
    _add_agent_args(p_loop)
    p_loop.add_argument("--iterations", type=int, default=3)
    p_loop.add_argument("--refactor-cmd", type=str, default=None,
                        help="shell command for the refactoring agent")
    return parser


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--family", action="append", default=[],
                   help="restrict to a task family (repeatable)")
    p.add_argument("--agent-cmd", type=str, default=None,
                   help="shell command for the evaluation agent (run in sandbox)")
    p.add_argument("--mutation-json", type=Path, default=None,
                   help="JSON file with mutation/fault evidence for T")


# --- Command handlers -------------------------------------------------------

def cmd_analyze(args: argparse.Namespace, config: EscConfig) -> int:
    analysis = analyze_static(args.root, config)
    if args.json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(render_static_analysis(analysis))
    return 0


def cmd_score(args: argparse.Namespace, config: EscConfig) -> int:
    values = _load_subscores(args)
    esc, effective = compute_esc_partial(values, config.weights)
    print(f"ESC = {esc:.2f} / 100")
    print(f"effective weights: {json.dumps({k: round(v, 3) for k, v in effective.items()})}")
    return 0


def cmd_evaluate(args: argparse.Namespace, config: EscConfig) -> int:
    seeds = _parse_seeds(args.seeds)
    families = select_families(default_families(), tuple(args.family))
    agent = _build_agent(args.agent_cmd)
    mutation = _load_mutation(args.mutation_json)
    result = evaluate_codebase(args.root, families, seeds, agent, config, mutation=mutation)
    if args.json:
        print(result_to_json(result))
    else:
        print(render_dashboard(result, config))
    return 0


def cmd_loop(args: argparse.Namespace, config: EscConfig) -> int:
    seeds = _parse_seeds(args.seeds)
    families = select_families(default_families(), tuple(args.family))
    eval_agent = _build_agent(args.agent_cmd)
    refactor_agent = _build_agent(args.refactor_cmd)
    mutation = _load_mutation(args.mutation_json)
    loop = ChampionCandidateLoop(
        source_root=args.root, eval_agent=eval_agent, refactor_agent=refactor_agent,
        config=config, families=families, seeds=seeds, mutation=mutation,
    )
    try:
        history = loop.run(args.iterations)
        print(render_loop(history))
    finally:
        loop.cleanup()
    return 0


# --- Helpers ----------------------------------------------------------------

def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(s) for s in raw.split(",") if s.strip())


def _build_agent(command: str | None):
    if not command:
        return ScriptedAgentRunner(name="scripted-noop")
    return SubprocessAgentRunner(SubprocessConfig(command=tuple(shlex.split(command))))


def _load_mutation(path: Path | None) -> MutationResult | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MutationResult(**data)


def _load_subscores(args: argparse.Namespace) -> dict[str, float | None]:
    keys = ("correctness", "edit_efficiency", "understandability",
            "locality", "testability", "static_quality")
    if args.subscores:
        data = json.loads(args.subscores.read_text(encoding="utf-8"))
        return {k: data.get(k) for k in keys}
    flags = (args.c, args.e, args.u, args.l, args.t, args.s)
    return dict(zip(keys, flags))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = EscConfig()
    handlers = {
        "analyze": cmd_analyze,
        "score": cmd_score,
        "evaluate": cmd_evaluate,
        "loop": cmd_loop,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
