from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Iterable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import typer
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

def _run_step_worker(ctx: Ctx, step: Step, part: Optional[Dict]) -> bool:
    """Top-level worker wrapper so ProcessPoolExecutor can pickle it."""
    return run_step(ctx, step, partition=part)
    
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

    chatty = bool(ctx.params.get("verbose", True))
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

        if ctx.only and name not in ctx.only:
            continue

        # -------------------------
        # PRE-TARGET VALIDATION
        # -------------------------
        if not reached:
            if name == ctx.skip_to:
                reached = True
            else:
                if step.is_partitioned:
                    assert partitions and len(partitions) > 0, "No partitions available for partitioned step"
                    for part in partitions:
                        label = f"{name}[{part.get('id','?')}]"
                        if _outs_exist(step, part):
                            skipped.append(label)
                        else:
                            typer.echo(f"[dag] → {name} (pre-target, materializing)")
                            ran = run_step(ctx, step, partition=part)
                            (executed if ran else skipped).append(label)
                else:
                    if _outs_exist(step, None):
                        skipped.append(name)
                    else:
                        typer.echo(f"[dag] → {name} (pre-target, materializing)")
                        ran = run_step(ctx, step, partition=None)
                        (executed if ran else skipped).append(name)
                continue

        # -------------------------
        # NORMAL EXECUTION
        # -------------------------
        if step.is_partitioned:
            assert partitions is not None and len(partitions) > 0, "No partitions available for partitioned step"
            if chatty:
                typer.echo(f"[dag] → {name} (n_parts={len(partitions)})")

            # Canary: run first N parts serially to fail fast on misconfig
            canary_n = int(os.environ.get("AMM_CANARY_N", "0"))
            if canary_n > 0 and name == ctx.skip_to:
                test_parts = partitions[:min(canary_n, len(partitions))]
                ok = 0
                for part in test_parts:
                    label = f"{name}[{part.get('id','?')}]"
                    try:
                        ran = run_step(ctx, step, partition=part)
                        (executed if ran else skipped).append(label)
                        if ran:
                            ok += 1
                    except Exception as e:
                        raise RuntimeError(f"{name} canary failed on {label}") from e
                if ok == 0:
                    raise RuntimeError(f"{name} canary: 0/{len(test_parts)} produced output; aborting.")

            max_workers = int(ctx.params.get("threads", os.cpu_count() or 1))
            max_workers = max(1, min(max_workers, len(partitions)))

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_step_worker, ctx, step, part): part for part in partitions}
                done = 0
                for fut in as_completed(futs):
                    part = futs[fut]
                    label = f"{name}[{part.get('id','?')}]"
                    try:
                        ran = fut.result()
                        (executed if ran else skipped).append(label)
                    except Exception as e:
                        raise RuntimeError(f"{name} failed on {label}: {e}") from e
                    finally:
                        done += 1
                        if chatty and (done % 50 == 0 or done == len(futs)):
                            typer.echo(f"[dag]   {name}: {done}/{len(futs)}")
        else:
            if chatty:
                typer.echo(f"[dag] → {name}")
            ran = run_step(ctx, step, partition=None)
            (executed if ran else skipped).append(name)

    return {"executed": executed, "skipped": skipped}
