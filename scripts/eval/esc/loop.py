"""The stochastic champion/candidate iteration procedure.

    champion = current codebase
    repeat:
        candidate = refactor(champion, random hotspot)
        if not hard_gates_pass(candidate): reject
        evaluate champion and candidate on the SAME tasks and seeds
        if candidate has a meaningful ESC gain and no major subscore regresses:
            champion = candidate
        else: reject

Safeguards from the design are enforced here: paired evaluation, a *separate*
evaluation agent from the refactoring agent (no self-grading), bounded patch
size (reject large unexplained rewrites), and a per-subscore regression guard.
The real workspace is never mutated — the loop operates entirely on copies.
"""

from __future__ import annotations

import random
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from esc.agents.base import AgentRunner, RefactorRequest
from esc.agents.subprocess_runner import git_patch_stats
from esc.config import EscConfig
from esc.evaluate import evaluate_codebase_detailed
from esc.gates import GateInputs, evaluate_gates
from esc.model import EscResult, GateReport, MutationResult, SubScores, TaskRunResult
from esc.stats import align_pairs, paired_bootstrap
from esc.tasks.base import TaskEnvironment
from esc.tasks.registry import TaskFamily, all_change_tasks, default_families
from esc.workspace import copy_codebase

CandidatePrecheck = Callable[[Path, TaskEnvironment, EscConfig], GateReport]


def default_precheck(root: Path, env: TaskEnvironment, config: EscConfig) -> GateReport:
    """A cheap CPU-only hard gate on the candidate codebase itself."""
    abi_ok = env.run_python(root, ("-m", "trainer.contracts", "--check")).ok
    import_ok = env.run_python(root, ("-c", "import trainer.config_types")).ok
    inputs = GateInputs(
        existing_tests_pass=import_ok,
        max_numeric_deviation=0.0,
        perf_ratio=1.0,
        abi_stable=abi_ok,
        deterministic=True,
        static_blockers=0,
        mutation_score=0.0,
    )
    return evaluate_gates(inputs, config.gates)


# --- Decision + history records ---------------------------------------------

@dataclass(frozen=True)
class LoopDecision:
    accept: bool
    reason: str
    esc_gain: float
    confidence: float
    patch_files: int
    patch_lines: int
    subscore_regressions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopIteration:
    index: int
    hotspot: str
    accepted: bool
    champion_esc: float
    candidate_esc: float
    decision: LoopDecision


@dataclass
class LoopHistory:
    iterations: list[LoopIteration]
    final_esc: float
    champion_root: Path


def _subscore_regressions(champion: SubScores, candidate: SubScores,
                          tolerance: float) -> dict[str, float]:
    regressions: dict[str, float] = {}
    for (label, champ_v), (_l, cand_v) in zip(champion.as_ordered(), candidate.as_ordered()):
        drop = champ_v - cand_v
        if drop > tolerance:
            regressions[label] = drop
    return regressions


class ChampionCandidateLoop:
    """Runs the champion/candidate procedure over copies of a source codebase."""

    def __init__(
        self,
        source_root: Path,
        eval_agent: AgentRunner,
        refactor_agent: AgentRunner,
        config: EscConfig,
        families: Sequence[TaskFamily] | None = None,
        seeds: Sequence[int] = (0, 1, 2),
        env: TaskEnvironment | None = None,
        hotspots: Sequence[str] | None = None,
        mutation: MutationResult | None = None,
        precheck: CandidatePrecheck = default_precheck,
        rng: random.Random | None = None,
    ) -> None:
        self.source_root = source_root
        self.eval_agent = eval_agent
        self.refactor_agent = refactor_agent
        self.config = config
        self.families = tuple(families) if families is not None else default_families()
        self.seeds = tuple(seeds)
        self.env = env or TaskEnvironment()
        self.hotspots = tuple(hotspots) if hotspots else tuple(c.name for c in config.concepts)
        self.mutation = mutation
        self.precheck = precheck
        self.rng = rng or random.Random(0)
        self._baselines = {t.name: t.baseline for t in all_change_tasks(self.families)}
        self._workdir = Path(tempfile.mkdtemp(prefix="esc_loop_"))

    # -- lifecycle -----------------------------------------------------------

    def cleanup(self) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)

    def run(self, iterations: int) -> LoopHistory:
        champion_root = copy_codebase(self.source_root, self._workdir / "champion", self.config)
        champion_result, champion_runs, _ = self._evaluate(champion_root)

        history: list[LoopIteration] = []
        for i in range(iterations):
            hotspot = self.rng.choice(self.hotspots)
            outcome = self._propose_and_judge(i, hotspot, champion_root,
                                              champion_result, champion_runs)
            history.append(outcome.iteration)
            if outcome.iteration.accepted:
                self._retire(champion_root)
                champion_root = outcome.candidate_root
                champion_result = outcome.candidate_result
                champion_runs = outcome.candidate_runs
            elif outcome.candidate_root is not None:
                shutil.rmtree(outcome.candidate_root, ignore_errors=True)

        return LoopHistory(history, champion_result.esc, champion_root)

    # -- internals -----------------------------------------------------------

    def _evaluate(self, root: Path) -> tuple[EscResult, list[TaskRunResult], list]:
        return evaluate_codebase_detailed(
            root, self.families, self.seeds, self.eval_agent, self.config,
            self.env, self.mutation,
        )

    def _propose_and_judge(
        self, index: int, hotspot: str, champion_root: Path,
        champion_result: EscResult, champion_runs: list[TaskRunResult],
    ) -> "_Proposal":
        candidate_root = copy_codebase(
            champion_root, self._workdir / f"candidate_{index}", self.config, init_git=True
        )
        refactor = self.refactor_agent.propose_refactor(
            RefactorRequest(candidate_root, hotspot, seed=index)
        )
        patch = git_patch_stats(candidate_root)

        reject = self._early_reject(refactor.changed, patch)
        if reject is not None:
            return _Proposal(_iteration(index, hotspot, False, champion_result.esc,
                                        champion_result.esc, reject, patch), candidate_root)

        gate = self.precheck(candidate_root, self.env, self.config)
        if not gate.passed:
            decision = LoopDecision(False, f"hard gate failed: {'; '.join(gate.failures)}",
                                    0.0, 0.0, patch.files_changed, patch.lines_changed)
            return _Proposal(_iteration(index, hotspot, False, champion_result.esc,
                                        champion_result.esc, decision, patch), candidate_root)

        candidate_result, candidate_runs, _ = self._evaluate(candidate_root)
        decision = self._decide(champion_result, candidate_result,
                                champion_runs, candidate_runs, patch)
        iteration = _iteration(index, hotspot, decision.accept, champion_result.esc,
                               candidate_result.esc, decision, patch)
        return _Proposal(iteration, candidate_root, candidate_result, candidate_runs)

    def _early_reject(self, changed: bool, patch) -> LoopDecision | None:
        if not changed:
            return LoopDecision(False, "no change proposed", 0.0, 0.0, 0, 0)
        controls = self.config.loop
        if patch.files_changed > controls.max_patch_files:
            return LoopDecision(False, f"patch touches {patch.files_changed} files "
                                f"(> {controls.max_patch_files})", 0.0, 0.0,
                                patch.files_changed, patch.lines_changed)
        if patch.lines_changed > controls.max_patch_lines:
            return LoopDecision(False, f"patch changes {patch.lines_changed} lines "
                                f"(> {controls.max_patch_lines})", 0.0, 0.0,
                                patch.files_changed, patch.lines_changed)
        return None

    def _decide(
        self, champion: EscResult, candidate: EscResult,
        champion_runs: list[TaskRunResult], candidate_runs: list[TaskRunResult], patch,
    ) -> LoopDecision:
        controls = self.config.loop
        esc_gain = candidate.esc - champion.esc
        regressions = _subscore_regressions(
            champion.subscores, candidate.subscores, controls.max_subscore_regression
        )
        pairs = align_pairs(champion_runs, candidate_runs, self._baselines, self.config.cost)
        boot = paired_bootstrap(pairs, controls.bootstrap_samples, self.rng)

        accept = (
            esc_gain >= controls.min_esc_gain
            and boot.confidence >= controls.bootstrap_confidence
            and not regressions
        )
        reason = self._reason(accept, esc_gain, boot.confidence, regressions, controls)
        return LoopDecision(accept, reason, esc_gain, boot.confidence,
                            patch.files_changed, patch.lines_changed, regressions)

    @staticmethod
    def _reason(accept, esc_gain, confidence, regressions, controls) -> str:
        if accept:
            return f"accepted: +{esc_gain:.2f} ESC at {confidence:.0%} confidence"
        if esc_gain < controls.min_esc_gain:
            return f"ESC gain {esc_gain:.2f} < {controls.min_esc_gain}"
        if confidence < controls.bootstrap_confidence:
            return f"confidence {confidence:.0%} < {controls.bootstrap_confidence:.0%}"
        return f"subscore regressions: {regressions}"

    def _retire(self, champion_root: Path) -> None:
        if champion_root != self.source_root:
            shutil.rmtree(champion_root, ignore_errors=True)


@dataclass
class _Proposal:
    iteration: LoopIteration
    candidate_root: Path | None
    candidate_result: EscResult | None = None
    candidate_runs: list[TaskRunResult] = field(default_factory=list)


def _iteration(index, hotspot, accepted, champion_esc, candidate_esc, decision, patch) -> LoopIteration:
    return LoopIteration(index, hotspot, accepted, champion_esc, candidate_esc, decision)
