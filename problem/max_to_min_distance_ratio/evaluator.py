import numpy as np
import time
import traceback
from typing import Any, Dict, Tuple

from utils import create_safe_execution_environment, get_code_context, enhance_error_message

N_POINTS = 16
FLOAT_TOLERANCE = 1e-8

def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return all pairwise Euclidean distances for i<j."""
    diffs = points[:, None, :] - points[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(points.shape[0], k=1)
    return dist_matrix[iu]

def max_to_min_distance_ratio(points: np.ndarray) -> float:
    """
    Calculate the maximum to minimum distance ratio of the given point set (the smaller the better).
    """
    dists = _pairwise_distances(points)

    if dists.size == 0:
        raise ValueError("At least two points are required")

    min_dist = float(np.min(dists))
    max_dist = float(np.max(dists))

    if min_dist <= FLOAT_TOLERANCE:
        raise ValueError("Minimum pairwise distance is too small (duplicate or near-duplicate points)")

    return max_dist / min_dist

def _validate_points(points: np.ndarray) -> Tuple[bool, bool]:
    """
    Validate if all points are within the unit square.

    Returns:
        (valid, out_of_bounds)
    """
    out_of_bounds = bool(np.any(points < -FLOAT_TOLERANCE) or np.any(points > 1.0 + FLOAT_TOLERANCE))
    return (not out_of_bounds), out_of_bounds

def evaluate(code: str, config: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate max-to-min distance ratio problem code.
    """
    config = config or {}

    # fitness dictionary: only contains the optimization objectives
    fitness: Dict[str, Any] = {
        "max_to_min_ratio": float("inf"),
    }

    # eval_info dictionary: used for prompt enhancement and observable information
    eval_info: Dict[str, Any] = {
        "validity": False,
        "min_pairwise_distance": 0.0,
        "max_pairwise_distance": 0.0,
        "execution_time": 0.0,
    }

    execution_time: float = 0.0
    start_time: float | None = None
    error_info: Dict[str, Any] = {}

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

        # Extract points
        if "points" in scope:
            points = np.array(scope["points"], dtype=float)
        elif callable(scope.get("construct_points")):
            points = np.array(scope["construct_points"](), dtype=float)
        else:
            raise ValueError("Code did not produce 'points' variable or construct_points() function")

        # Basic validity check
        if np.isnan(points).any() or np.isinf(points).any():
            raise ValueError("Points contain NaN or infinite values")
        if points.shape != (N_POINTS, 2):
            raise ValueError(f"Invalid points shape: {points.shape}, expected ({N_POINTS}, 2)")

        valid, out_of_bounds = _validate_points(points)
        if not valid:
            eval_info["validity"] = False
            eval_info["execution_time"] = float(execution_time)
            error_info = {
                "type": "ConstraintViolation",
                "message": "Points are outside unit square constraints.",
                "details": {
                    "out_of_bounds": out_of_bounds,
                    "num_points": int(len(points)),
                },
            }
            return fitness, eval_info, error_info

        # Additional statistical information
        pairwise_dists = _pairwise_distances(points)
        min_dist = float(np.min(pairwise_dists))
        max_dist = float(np.max(pairwise_dists))
        if min_dist <= FLOAT_TOLERANCE:
            raise ValueError("Minimum pairwise distance is too small (duplicate or near-duplicate points)")
        ratio = max_dist / min_dist
        execution_time = time.time() - start_time

        eval_info["validity"] = bool(valid)
        eval_info["min_pairwise_distance"] = min_dist
        eval_info["max_pairwise_distance"] = max_dist
        eval_info["execution_time"] = float(execution_time)

        fitness["max_to_min_ratio"] = float(ratio)

    except Exception as e:
        if start_time is not None and execution_time <= 0.0:
            execution_time = time.time() - start_time
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

        fitness["max_to_min_ratio"] = float("inf")
        eval_info["validity"] = False
        eval_info["execution_time"] = float(execution_time)
        eval_info["min_pairwise_distance"] = 0.0
        eval_info["max_pairwise_distance"] = 0.0

        if config.get("verbose", False):
            print(f"Execution error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

    return fitness, eval_info, error_info