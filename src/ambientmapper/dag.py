from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Iterable, Any
from .sentinels import (
    sentinel_path, load_sentinel, write_sentinel,
    make_payload, fingerprint, read_generation, bump_generation, sentinel_ok,
)

@dataclass
class Ctx:
    cfg: Dict[str, Any]
    params: Dict[str, Any]
    dirs: Dict[str, Path]
    version: str
    resume: bool = True
    force: List[str] = field(default_factory=list)
    skip_to: str = ""
    only: List[str] = field(default_factory=list)
    generation_id: int = 0

@dataclass
class Step:
    name: str
    requires: List[str]
    is_partitioned: bool
    inputs_fn: Callable[[Ctx, Optional[Dict]], List[Path]]
    outputs_fn: Callable[[Ctx, Optional[Dict]], List[Path]]
    runner_fn: Callable[[Ctx, Optional[Dict]], None]
    bump_generation: bool = False

def topo_sort(steps: Dict[str, Step]) -> List[str]:
    order = []
    temp = set()
    perm = set()

    def visit(n: str):
        if n in perm: return
        if n in temp: raise RuntimeError(f"Cycle at {n}")
        temp.add(n)
        for r in steps[n].requires:
            visit(r)
        perm.add(n); temp.remove(n); order.append(n)

    for name in steps:
        visit(name)
    return order

def should_run_this_step(ctx: Ctx, step: Step) -> bool:
    if ctx.only and step.name not in ctx.only:
        return False
    return True

def run_step(ctx: Ctx, step: Step, partition: Optional[Dict] = None) -> bool:
    ins = step.inputs_fn(ctx, partition)
    outs = step.outputs_fn(ctx, partition)

    chunk_id = partition.get("id") if (partition and "id" in partition) else None
    s_name = step.name if not step.is_partitioned else f"{step.name}"
    s_path = sentinel_path(ctx.dirs["root"], s_name, chunk_id if step.is_partitioned else None)

    fp = fingerprint(ctx.cfg, ctx.params, ins, ctx.generation_id)

    if ctx.resume and sentinel_ok(load_sentinel(s_path), fp, outs):
        return False

    if ctx.skip_to and step.name != ctx.skip_to:
        missing = [p for p in outs if not Path(p).exists()]
        if not missing:
            return False

    started = time.time()
    step.runner_fn(ctx, partition)
    payload = make_payload(step.name, fp, ins, outs, os.getpid(), started, ctx.version)
    write_sentinel(s_path, payload)

    if step.bump_generation and not step.is_partitioned:
        ctx.generation_id = bump_generation(ctx.dirs["root"], reason=f"{step.name} completed")
    return True

def run_dag(ctx: Ctx, steps: Dict[str, Step], partitions: List[Dict] | None) -> Dict[str, Any]:
    """
    Execute steps in topological order.

    Behavior with ctx.skip_to:
      - For steps BEFORE ctx.skip_to: "validate-only" mode
          * If all expected outputs exist and are non-empty -> skip
          * Otherwise -> run the step (to materialize missing outputs)
      - Starting AT ctx.skip_to (and thereafter): normal execution (resume via sentinels)
    """
    order = topo_sort(steps)
    executed: List[str] = []
    skipped: List[str] = []

    # If no skip_to is provided, we start in "reached" (normal) mode.
    reached = not bool(ctx.skip_to)

    def _outs_exist(step: Step, part: Optional[Dict]) -> bool:
        outs = step.outputs_fn(ctx, part)
        for o in outs:
            p = Path(o)
            if (not p.exists()) or (p.stat().st_size == 0):
                return False
        return True

    for name in order:
        step = steps[name]

        # respect --only-steps if provided
        if ctx.only and name not in ctx.only:
            continue

        # PRE-TARGET VALIDATION PASS
        if not reached:
            if name == ctx.skip_to:
                # flip into normal execution from here on
                reached = True
            else:
                # validate-only for pre-target steps
                if step.is_partitioned:
                    assert partitions is not None and len(partitions) > 0, "No partitions available for partitioned step"
                    for part in partitions:
                        label = f"{name}[{part.get('id','?')}]"
                        if _outs_exist(step, part):
                            skipped.append(label)
                        else:
                            ran = run_step(ctx, step, partition=part)
                            (executed if ran else skipped).append(label)
                else:
                    if _outs_exist(step, None):
                        skipped.append(name)
                    else:
                        ran = run_step(ctx, step, partition=None)
                        (executed if ran else skipped).append(name)
                continue  # move to next step

        # NORMAL EXECUTION (from target onward)
        if step.is_partitioned:
            assert partitions is not None and len(partitions) > 0, "No partitions available for partitioned step"
            for part in partitions:
                label = f"{name}[{part.get('id','?')}]"
                ran = run_step(ctx, step, partition=part)
                (executed if ran else skipped).append(label)
        else:
            ran = run_step(ctx, step, partition=None)
            (executed if ran else skipped).append(name)

    return {"executed": executed, "skipped": skipped}
