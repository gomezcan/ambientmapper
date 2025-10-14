from __future__ import annotations
import os, json, time, hashlib, contextlib
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Any

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _sha256_json(obj: object) -> str:
    return _sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())

def _inputs_meta(paths: Iterable[Path]) -> List[Tuple[str, int, float]]:
    meta = []
    for p in map(Path, paths):
        st = p.stat()
        meta.append((str(p), st.st_size, st.st_mtime))
    return meta

def fingerprint(cfg: Dict[str, Any], params: Dict[str, Any], inputs: Iterable[Path], upstream_generation: int) -> Dict[str, str]:
    return {
        "config_sha256": _sha256_json(cfg),
        "params_sha256": _sha256_json(params),
        "inputs_sha256": _sha256_json(_inputs_meta(inputs)),
        "upstream_generation": str(upstream_generation),
    }

def _gen_path(root: Path) -> Path:
    return Path(root) / "final" / "generation.json"

def read_generation(root: Path) -> int:
    p = _gen_path(root)
    if not p.exists():
        return 0
    try:
        with p.open("r") as f:
            data = json.load(f)
        return int(data.get("id", 0))
    except Exception:
        return 0

def bump_generation(root: Path, reason: str = "") -> int:
    root = Path(root)
    (root / "final").mkdir(parents=True, exist_ok=True)
    p = _gen_path(root)
    now = int(time.time())
    if p.exists():
        with p.open("r") as f:
            data = json.load(f)
    else:
        data = {"id": 0, "history": []}
    data["id"] = int(data.get("id", 0)) + 1
    data.setdefault("history", []).append({"id": data["id"], "ts": now, "reason": reason})
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)
    return data["id"]

def _sent_base(root: Path) -> Path:
    return Path(root) / "_sentinels"

def sentinel_path(root: Path, step: str, chunk: str | None = None) -> Path:
    base = _sent_base(root) / step
    if chunk is None:
        base = _sent_base(root)
        out = base / f"{step}.ok.json"
    else:
        d = _sent_base(root) / step
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"{chunk}.ok.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def load_sentinel(path: Path) -> Dict | None:
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def write_sentinel(path: Path, payload: Dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)

def make_payload(step: str, fingerprint: Dict[str, str], inputs, outputs, pid: int, started: float, version: str, note: str = "") -> Dict:
    return {
        "step": step,
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "ambientmapper_version": version,
        "fingerprint": fingerprint,
        "inputs": list(map(str, inputs)),
        "outputs": list(map(str, outputs)),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "pid": pid,
        "duration_sec": round(time.time() - started, 3),
        "note": note,
    }

def sentinel_ok(existing: Dict | None, expected_fp: Dict[str, str], outputs: Iterable[Path]) -> bool:
    if not existing:
        return False
    f = existing.get("fingerprint", {})
    ok = (
        f.get("config_sha256") == expected_fp["config_sha256"] and
        f.get("params_sha256") == expected_fp["params_sha256"] and
        f.get("inputs_sha256") == expected_fp["inputs_sha256"] and
        f.get("upstream_generation") == expected_fp["upstream_generation"] and
        existing.get("status") == "ok"
    )
    if not ok:
        return False
    for p in outputs:
        if (not p.exists()) or (p.stat().st_size == 0):
            return False
    return True

def remove_step_sentinels(root: Path, step_prefix: str) -> int:
    removed = 0
    base = _sent_base(root)
    if not base.exists():
        return 0
    for dirpath, _, files in os.walk(base):
        for fn in files:
            if not fn.endswith(".ok.json"):
                continue
            full = Path(dirpath) / fn
            if step_prefix in str(full):
                try:
                    full.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed
