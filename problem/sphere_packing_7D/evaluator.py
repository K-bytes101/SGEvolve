"""
Evaluator for 7D sphere packing example (n=26) with improved timeout handling
"""
import math
import numpy as np
import time
import traceback
from typing import Dict, Any, Optional

from utils import create_safe_execution_environment, get_code_context, enhance_error_message
# Problem constants
DIMENSION = 7
# Placeholder target value (3D value was copied from sphere_packing and is not applicable in 7D).
TARGET_VALUE = 0.0


def validate_packing(centers, radii, bounds=(0.0, 1.0), tol: float = 1e-6):
    """
    Validate the validity, check if there is any overlap or out-of-bounds

    Args:
        centers: The center coordinates of the spheres (x1, ..., x7), (n, 7)
        radii: The radius of the spheres r, (n,)
        bounds: The boundary of the 7D unit hypercube
        tol:
    """
    n = centers.shape[0]

    min_bound, max_bound = bounds
    has_overlaps = False
    out_of_bounds = False

    # Check if there is any out-of-bounds
    for i in range(n):
        point = centers[i]
        r = radii[i]
        if np.any(point - r < min_bound - tol) or np.any(point + r > max_bound + tol):
            out_of_bounds = True
            break

    # Check if there is any overlap
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


def evaluate(code: str, config: Dict[str, Any]):
    config = config or {}

    # Initialize the return value
    fitness: Dict[str, Any] = {
        "sum_of_radii": 0.0,
    }

    eval_info: Dict[str, Any] = {
        "validity": False,
        "packing_density": 0.0,
        "execution_time": 0.0,
    }

    execution_time: float = 0.0
    error_info: Dict[str, Any] = {}

    # Construct the safe execution environment, and inject numpy
    safe_globals = create_safe_execution_environment()
    safe_globals["np"] = np

    try:
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

        start_time = time.time()
        scope: Dict[str, Any] = dict(safe_globals)
        try:
            exec(compiled_code, scope, scope)
        except Exception as exec_error:
            enhanced_msg = enhance_error_message(exec_error, code, code_id)
            raise type(exec_error)(enhanced_msg) from exec_error

        execution_time = time.time() - start_time

        # Extract variables and perform basic validation
        if callable(scope.get("construct_packing")):
            centers, radii = scope["construct_packing"]()
        elif "centers" in scope and "radii" in scope:
            centers = np.array(scope["centers"], dtype=float)
            radii = np.array(scope["radii"], dtype=float)
        else:
            raise ValueError("Code did not produce 'centers' and 'radii' variables")

        # Check the validity of the return value
        if np.isnan(centers).any() and np.isnan(radii).any():
            raise ValueError("NaN values detected in centers and radii")
        elif np.isnan(centers).any():
            raise ValueError("NaN values detected in centers")
        elif np.isnan(radii).any():
            raise ValueError("NaN values detected in radii")

        # The radius must be positive
        if np.any(radii <= 0):
            raise ValueError("All radii must be positive")

        # Check the shape of the return value
        if centers.shape != (26, DIMENSION) and radii.shape != (26,):
            raise ValueError(
                f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, {DIMENSION}) and (26,)"
            )
        elif centers.shape != (26, DIMENSION):
            raise ValueError(f"Invalid centers shape: {centers.shape}, expected (26, {DIMENSION})")
        elif radii.shape != (26,):
            raise ValueError(f"Invalid radii shape: {radii.shape}, expected (26,)")

        # Check the geometric constraints
        valid, has_overlaps, out_of_bounds = validate_packing(centers, radii)

        # Calculate the metrics
        raw_sum_radii = float(np.sum(radii))
        sum_radii = raw_sum_radii if valid else 0.0

        # 7D hypersphere volume: V_d(r) = pi^(d/2) / Gamma(d/2 + 1) * r^d
        unit_ball_coeff = math.pi ** (DIMENSION / 2.0) / math.gamma(DIMENSION / 2.0 + 1.0)
        total_sphere_volume = np.sum(unit_ball_coeff * radii ** DIMENSION)
        total_cube_volume = 1.0  # The volume of the unit hypercube
        packing_density = float(total_sphere_volume / total_cube_volume)

        # Record the evaluation information
        eval_info["validity"] = bool(valid)
        eval_info["execution_time"] = execution_time
        if valid:
            eval_info["packing_density"] = packing_density
        else:
            eval_info["packing_density"] = 0.0

            error_info = {
                "type": "ConstraintViolation",
                "message": "Packing violates geometric constraints (overlap and/or out-of-bounds).",
                "details": {
                    "has_overlaps": has_overlaps,
                    "out_of_bounds": out_of_bounds,
                    "packing_density": packing_density,
                },
            }

        fitness["sum_of_radii"] = sum_radii

    except Exception as e:
        fitness["sum_of_radii"] = 0.0

        eval_info = {
            "validity": False,
            "packing_density": 0.0,
            "execution_time": 0.0,
        }

        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

        if config.get("verbose", False):
            print(f"Execution error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

    return fitness, eval_info, error_info