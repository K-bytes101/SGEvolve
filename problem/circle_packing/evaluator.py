import numpy as np
import time
import traceback
from typing import Any, Dict, Tuple
from utils import create_safe_execution_environment, get_code_context, enhance_error_message


def validate_packing(centers: np.ndarray, radii: np.ndarray, bounds=(0.0, 1.0), tol: float = 1e-6) -> Tuple[bool, bool, bool]:

    n = centers.shape[0]
    if n != radii.shape[0]:
        return False, False, False

    min_bound, max_bound = bounds
    has_overlaps = False
    out_of_bounds = False

    # check the boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < min_bound - tol or x + r > max_bound + tol or y - r < min_bound - tol or y + r > max_bound + tol:
            out_of_bounds = True
            break

    # check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - tol:
                has_overlaps = True
                break
        if has_overlaps:
            break

    valid = (not has_overlaps) and (not out_of_bounds)
    return valid, has_overlaps, out_of_bounds


def evaluate(code: str, config: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate circle-packing code.

    Agreement: 
    - Return three information:
        1) fitness: the dictionary of metrics for multi-objective optimization (here only contains sum_of_radii):
        2) eval_info: detailed intermediate calculation information (packing_density and execution_time), for LLM prompt:
        3) error_info: error or constraint violation information (empty dictionary in normal case).
    """
    config = config or {}
    bounds = tuple(config.get('bounds', (0.0, 1.0)))  # (min, max)

    # fitness dictionary: only keep the metrics for multi-objective optimization
    # corresponding to the name in config.prompt.objectives
    fitness: Dict[str, Any] = {
        "sum_of_radii": 0.0,
    }

    # eval_info dictionary: for LLM prompt
    eval_info: Dict[str, Any] = {
        "validity": False,
        "packing_density": 0.0,
        "execution_time": 0.0,
    }

    execution_time: float = 0.0
    error_info: Dict[str, Any] = {}

    # construct safe execution environment, and inject numpy (generic做法：问题 evaluator 自己决定需要哪些库)
    safe_globals = create_safe_execution_environment()
    safe_globals["np"] = np

    try:
        start_time = time.time()
        code_id = "<user_code>"

        try:
            compiled_code = compile(code, code_id, "exec")
        except SyntaxError as e:
            enhanced_msg = (
                f"Syntax error in code at line {e.lineno}: {e.msg}\n"
                f"Code snippet around line {e.lineno}:\n"
                f"{get_code_context(code, e.lineno)}"
            )
            raise SyntaxError(enhanced_msg) from e

        scope: Dict[str, Any] = dict(safe_globals)
        try:
            exec(compiled_code, scope, scope)
        except Exception as exec_error:
            enhanced_msg = enhance_error_message(exec_error, code, code_id)
            raise type(exec_error)(enhanced_msg) from exec_error

        execution_time = time.time() - start_time

        # ---------- extract variables from execution scope ----------
        if callable(scope.get("search_algorithm")):
            centers, radii = scope["search_algorithm"]()
        else:
            centers = np.array(scope["centers"], dtype=float)
            radii = np.array(scope["radii"], dtype=float)

        if centers is None or radii is None:
            raise ValueError('Code did not produce centers or radii via search_algorithm() or "centers" and "radii" variables')

        # check the validity of the return values
        if np.isnan(centers).any() and np.isnan(radii).any():
            raise ValueError("NaN values detected in centers and radii")
        elif np.isnan(centers).any():
            raise ValueError("NaN values detected in centers")
        elif np.isnan(radii).any():
            raise ValueError("NaN values detected in radii")

        # check the shape of the return values
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"Invalid centers shape: {centers.shape}, expected (n, 2)")
        if radii.ndim != 1:
            raise ValueError(f"Invalid radii shape: {radii.shape}, expected (n,)")
        if len(centers) != len(radii):
            raise ValueError(f"Mismatch: {len(centers)} centers but {len(radii)} radii")
        if len(centers) == 0:
            raise ValueError("No circles provided")

        # the radius must be positive
        if np.any(radii <= 0):
            raise ValueError("All radii must be positive")

        # ---------- geometric metrics ----------
        # only check the validity and basic statistics
        valid, has_overlaps, out_of_bounds = validate_packing(centers, radii, bounds=bounds)

        raw_sum_radii = float(np.sum(radii))
        sum_radii = raw_sum_radii if valid else 0.0

        total_circle_area = float(np.sum(np.pi * radii ** 2))
        total_area = float((bounds[1] - bounds[0]) ** 2)
        packing_density = total_circle_area / total_area if total_area > 0 else 0.0

        # record evaluation information
        eval_info["validity"] = bool(valid)
        if valid:
            eval_info["packing_density"] = float(packing_density)
            eval_info["execution_time"] = float(execution_time)
        else:
            eval_info["packing_density"] = 0.0
            eval_info["execution_time"] = float(execution_time)

            error_info = {
                "type": "ConstraintViolation",
                "message": "Packing violates geometric constraints (overlap and/or out-of-bounds).",
                "details": {
                    "has_overlaps": has_overlaps,
                    "out_of_bounds": out_of_bounds,
                    "num_circles": len(centers),
                },
            }

        # fill the result (only write the target values)
        fitness["sum_of_radii"] = sum_radii

    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

        fitness["sum_of_radii"] = 0.0
        eval_info["validity"] = False
        eval_info["execution_time"] = execution_time
        eval_info["packing_density"] = 0.0

        if config.get("verbose", False):
            print(f"Execution error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

    return fitness, eval_info, error_info