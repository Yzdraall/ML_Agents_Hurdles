"""Convert PyTorch 2.6+ dynamo ONNX exports into Unity Sentis-compatible ONNX files.

Unity Sentis (com.unity.ai.inference 2.x) fails on recent mlagents-learn exports because:
1. Weights stored in external .onnx.data files may be ordered arbitrarily; Sentis'
   ONNXConstantsLoader reads ExternalData lengths without seeking to tensor offsets.
2. Unused 0-width inputs (action_masks, recurrent_in) remain in graph.input.
3. Static batch=1 shapes appear in value_info and action outputs despite dynamic axes.
4. Opset 18 / IR 10 exceed Sentis 2.5's baseline opset 15 target.

Usage:
    python scripts/fix_onnx_for_unity.py --from-run hurdles_v3
    python scripts/fix_onnx_for_unity.py [--check] [path/to/model.onnx ...]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, checker, external_data_helper, helper, numpy_helper, shape_inference

TARGET_IR_VERSION = 8
TARGET_OPSET = 15
BATCH_DIM_NAME = "batch"
DEFAULT_BEHAVIORS = ("HurdleWalker", "HurdleCrawler")
DYNAMIC_BATCH_OUTPUTS = {
    "continuous_actions",
    "deterministic_continuous_actions",
    "discrete_actions",
    "deterministic_discrete_actions",
    "recurrent_out",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_brains_dir() -> Path:
    return _repo_root() / "Assets" / "Models" / "Brains"


def _repo_results_dir() -> Path:
    return _repo_root() / "results"


def _set_dynamic_batch(value_info: onnx.ValueInfoProto) -> None:
    shape = value_info.type.tensor_type.shape
    if not shape.dim:
        return
    dim0 = shape.dim[0]
    dim0.ClearField("dim_value")
    dim0.dim_param = BATCH_DIM_NAME


def _inline_initializers(model: onnx.ModelProto, ext_base_dir: Path | None = None) -> None:
    """Convert all loaded initializers and Constant attributes into inline TensorProto payloads."""
    graph = model.graph
    inlined_inits = [
        numpy_helper.from_array(numpy_helper.to_array(init), name=init.name)
        for init in graph.initializer
    ]
    del graph.initializer[:]
    graph.initializer.extend(inlined_inits)

    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t") and attr.t.data_location == TensorProto.EXTERNAL:
                base = str(ext_base_dir) if ext_base_dir is not None else None
                arr = numpy_helper.to_array(attr.t, base_dir=base)
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=attr.t.name))


def _resolve_external_dir(locations: set[str], src_dir: Path, extra_dirs: Iterable[Path]) -> Path:
    """Find the directory holding all external .onnx.data files referenced by a model."""
    for candidate_dir in [src_dir, *extra_dirs]:
        if candidate_dir.is_dir() and all((candidate_dir / loc).is_file() for loc in locations):
            return candidate_dir

    results_dir = _repo_results_dir()
    if results_dir.is_dir():
        for loc in locations:
            matches = sorted(results_dir.rglob(Path(loc).name))
            for match in matches:
                candidate_dir = match.parent
                if all((candidate_dir / item).is_file() for item in locations):
                    return candidate_dir

    missing = ", ".join(sorted(loc for loc in locations if not (src_dir / loc).is_file()))
    raise FileNotFoundError(
        f"External ONNX weights not found in '{src_dir}' or '{results_dir}': {missing}. "
        "Copy the matching .onnx.data file alongside the .onnx or use --from-run <run-id>."
    )


def load_inlined_raw_model(src_path: Path, extra_dirs: Iterable[Path] = ()) -> onnx.ModelProto:
    """Load an ONNX model, resolving external .onnx.data files from src_dir or results/."""
    model = onnx.load(str(src_path), load_external_data=False)
    ext_locations = {
        entry.value
        for init in model.graph.initializer
        if init.data_location == TensorProto.EXTERNAL
        for entry in init.external_data
        if entry.key == "location"
    }
    ext_dir = _resolve_external_dir(ext_locations, src_path.parent, extra_dirs) if ext_locations else src_path.parent
    if ext_locations:
        external_data_helper.load_external_data_for_model(model, str(ext_dir))
    _inline_initializers(model, ext_base_dir=ext_dir)
    return model


def convert_model(inlined_model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a Sentis-compatible inline opset-15 copy of an already inlined ML-Agents ONNX model."""
    converted = onnx.ModelProto()
    converted.CopyFrom(inlined_model)
    graph = converted.graph

    # 1. Remove unused zero-width inputs left behind by dynamo tracing (e.g. action_masks, recurrent_in).
    consumed_names = {inp for node in graph.node for inp in node.input if inp}
    consumed_names.update(out.name for out in graph.output)
    kept_inputs = []
    for inp in graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim if d.HasField("dim_value")]
        is_zero_width = 0 in dims
        if is_zero_width and inp.name not in consumed_names:
            continue
        kept_inputs.append(inp)
    del graph.input[:]
    graph.input.extend(kept_inputs)

    # 2. Restore consistent symbolic 'batch' dimension on observation inputs and action outputs.
    for inp in graph.input:
        if inp.name.startswith("obs_") or inp.name in {"vector_observation", "prev_action", "recurrent_in", "action_masks"}:
            _set_dynamic_batch(inp)

    for out in graph.output:
        if out.name in DYNAMIC_BATCH_OUTPUTS:
            _set_dynamic_batch(out)

    # 3. Drop stale static [1, ...] value_info annotations and re-infer shapes from the dynamic inputs.
    del graph.value_info[:]

    # 4. Normalize ONNX opset/IR to Sentis-supported baseline.
    converted.ir_version = TARGET_IR_VERSION
    if converted.opset_import:
        for opset in converted.opset_import:
            if opset.domain == "":
                opset.version = TARGET_OPSET
    else:
        converted.opset_import.extend([helper.make_opsetid("", TARGET_OPSET)])

    converted = shape_inference.infer_shapes(converted)
    checker.check_model(converted, full_check=True)
    return converted


def verify_onnxruntime(model_path: Path, reference_path: Path | None = None) -> dict[str, object]:
    """Validate batch=1 and batch=16 execution with ONNX Runtime and compare against reference."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    summary: dict[str, object] = {}

    for batch in (1, 16):
        feed = {}
        for inp in sess.get_inputs():
            shape = [batch if (d is None or isinstance(d, str)) else int(d) for d in inp.shape]
            feed[inp.name] = rng.standard_normal(shape, dtype=np.float32)

        output_names = [o.name for o in sess.get_outputs()]
        outputs = dict(zip(output_names, sess.run(output_names, feed)))

        for key in ("continuous_actions", "deterministic_continuous_actions", "discrete_actions", "deterministic_discrete_actions"):
            if key in outputs:
                arr = outputs[key]
                if arr.shape[0] != batch:
                    raise AssertionError(f"{model_path.name}: {key} returned batch {arr.shape[0]}, expected {batch}")
                if not np.all(np.isfinite(arr)):
                    raise AssertionError(f"{model_path.name}: {key} contains non-finite values at batch={batch}")

        if batch == 1:
            summary["version"] = int(outputs["version_number"][0]) if "version_number" in outputs else None
            summary["memory_size"] = int(outputs["memory_size"][0]) if "memory_size" in outputs else None
            if "continuous_action_output_shape" in outputs:
                summary["continuous_size"] = int(outputs["continuous_action_output_shape"][0])

            if reference_path is not None and "deterministic_continuous_actions" in outputs:
                ref_sess = ort.InferenceSession(str(reference_path), providers=["CPUExecutionProvider"])
                ref_feed = {i.name: feed[i.name] for i in ref_sess.get_inputs() if i.name in feed}
                for i in ref_sess.get_inputs():
                    if i.name not in ref_feed:
                        shape = [1 if (d is None or isinstance(d, str)) else int(d) for d in i.shape]
                        ref_feed[i.name] = np.zeros(shape, dtype=np.float32)
                ref_out = ref_sess.run(["deterministic_continuous_actions"], ref_feed)[0]
                max_diff = float(np.max(np.abs(outputs["deterministic_continuous_actions"] - ref_out)))
                summary["max_diff"] = max_diff
                if max_diff > 1e-5:
                    raise AssertionError(f"{model_path.name}: max diff {max_diff} exceeds tolerance")

    return summary


def check_sentis_compliance(model_path: Path) -> None:
    """Assert that a saved .onnx file meets Unity Sentis requirements without external files."""
    model = onnx.load(str(model_path), load_external_data=False)
    for init in model.graph.initializer:
        if init.data_location == TensorProto.EXTERNAL or init.external_data:
            raise AssertionError(f"{model_path.name}: initializer '{init.name}' still references external data")

    opsets = {op.domain: op.version for op in model.opset_import}
    if opsets.get("", 0) > TARGET_OPSET:
        raise AssertionError(f"{model_path.name}: opset {opsets.get('')} exceeds {TARGET_OPSET}")

    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim if d.HasField("dim_value")]
        if 0 in dims:
            raise AssertionError(f"{model_path.name}: unused 0-width input '{inp.name}' still present")

    for out in model.graph.output:
        if out.name in DYNAMIC_BATCH_OUTPUTS:
            dim0 = out.type.tensor_type.shape.dim[0]
            if not dim0.dim_param:
                raise AssertionError(f"{model_path.name}: output '{out.name}' has static batch dim {dim0.dim_value}")

    checker.check_model(model, full_check=True)


def _step_number(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def resolve_run_jobs(run_id: str) -> list[tuple[Path, Path, list[Path]]]:
    """Map exported brains in results/<run-id>/ to Assets/Models/Brains/<Behavior>.onnx."""
    run_dir = Path(run_id) if Path(run_id).is_dir() else _repo_results_dir() / run_id
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    brains_dir = _repo_brains_dir()
    brains_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path, list[Path]]] = []

    for behavior in DEFAULT_BEHAVIORS:
        final_onnx = run_dir / f"{behavior}.onnx"
        behavior_dir = run_dir / behavior
        extra_dirs = [d for d in (behavior_dir, run_dir) if d.is_dir()]

        src = final_onnx if final_onnx.is_file() else None
        if src is None:
            candidates = []
            for search_dir in extra_dirs:
                candidates.extend(search_dir.glob(f"{behavior}-*.onnx"))
            if candidates:
                src = max(candidates, key=_step_number)

        if src is not None:
            jobs.append((src, brains_dir / f"{behavior}.onnx", extra_dirs))

    if not jobs:
        raise SystemExit(f"No HurdleWalker/HurdleCrawler .onnx files found under {run_dir}")
    return jobs


def resolve_inplace_jobs(paths: Iterable[str]) -> list[tuple[Path, Path, list[Path]]]:
    """Resolve explicit .onnx paths/directories or default Assets/Models/Brains/*.onnx."""
    raw = list(paths)
    if not raw:
        files = sorted(p for p in _repo_brains_dir().glob("*.onnx") if p.is_file())
    else:
        files = []
        for item in raw:
            p = Path(item)
            files.extend(sorted(p.glob("*.onnx")) if p.is_dir() else [p])
    return [(p, p, []) for p in files]


def process_model(src_path: Path, dst_path: Path, extra_dirs: Iterable[Path]) -> None:
    """Convert src_path into a Sentis-compliant model at dst_path and verify exact parity."""
    tmp_ref = dst_path.with_suffix(".onnx.ref_tmp")
    inlined_orig = load_inlined_raw_model(src_path, extra_dirs=extra_dirs)
    fixed_model = convert_model(inlined_orig)

    try:
        onnx.save(inlined_orig, str(tmp_ref), save_as_external_data=False)
        onnx.save(fixed_model, str(dst_path), save_as_external_data=False)
        check_sentis_compliance(dst_path)
        stats = verify_onnxruntime(dst_path, reference_path=tmp_ref)
    finally:
        if tmp_ref.exists():
            tmp_ref.unlink()

    print(f"[FIXED] {src_path} -> {dst_path.name}: {dst_path.stat().st_size / (1024 * 1024):.2f} MB | {stats}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inline weights and fix ONNX exports for Unity Sentis.")
    parser.add_argument("models", nargs="*", help="Path(s) to .onnx files or directories (defaults to Assets/Models/Brains)")
    parser.add_argument("--from-run", metavar="RUN_ID", help="Convert results/<RUN_ID> brains directly into Assets/Models/Brains/")
    parser.add_argument("--check", action="store_true", help="Verify Sentis compliance and ONNX Runtime outputs without rewriting")
    args = parser.parse_args()

    if args.from_run and args.models:
        raise SystemExit("Pass either --from-run RUN_ID or explicit model paths, not both.")
    if args.from_run and args.check:
        raise SystemExit("--check verifies saved .onnx files and cannot be combined with --from-run.")

    jobs = resolve_run_jobs(args.from_run) if args.from_run else resolve_inplace_jobs(args.models)
    if not jobs:
        raise SystemExit("No .onnx files found.")

    for src_path, dst_path, extra_dirs in jobs:
        if not src_path.exists():
            raise SystemExit(f"File not found: {src_path}")

        if args.check:
            check_sentis_compliance(src_path)
            stats = verify_onnxruntime(src_path)
            print(f"[OK] {src_path.name}: {src_path.stat().st_size / (1024 * 1024):.2f} MB | {stats}")
        else:
            process_model(src_path, dst_path, extra_dirs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
