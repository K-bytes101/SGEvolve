import itertools
import time
import traceback
import numpy as np
from typing import Tuple, Dict, Any

from utils import create_safe_execution_environment, get_code_context, enhance_error_message
# The vertices of the target equilateral triangle
EDGE_LEN = 1.0
V1 = np.array([0.0, 0.0])
V2 = np.array([EDGE_LEN, 0.0])
V3 = np.array([EDGE_LEN / 2.0, EDGE_LEN * np.sqrt(3) / 2.0])
FLOAT_TOLERANCE = 1e-8

def get_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given its three vertices."""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) # type: ignore

def get_min_triangle_area(points: np.ndarray) -> float:
    """
    Calculates the minimum area of a triangle formed by any three points in the given set.
    """
    num_points = points.shape[0]
    if num_points < 3:
        return 0.0

    min_area = float('inf')
    
    # Iterate through all combinations of 3 points
    for p1, p2, p3 in itertools.combinations(points, 3):
        area = get_triangle_area(p1, p2, p3)
        if area < min_area:
            min_area = area
            
    return min_area

def validate_points(points: np.ndarray) -> Tuple[bool, bool]:
    """Checks if a point is outside the predefined equilateral triangle."""
    # Using barycentric coordinates to check if the point is inside
    # We can express the point as P = a*V1 + b*V2 + c*V3, where a+b+c=1.
    # The point is inside if a, b, c are all >= 0.
    # Since V1 is origin, we solve for P = b*V2 + c*V3
    out_of_triangle = False

    for point in points:
        try:
            A = np.column_stack((V2, V3))
            b_c = np.linalg.solve(A, point)
            b, c = b_c[0], b_c[1]
            a = 1 - b - c
            out_of_triangle = a < -FLOAT_TOLERANCE or b < -FLOAT_TOLERANCE or c < -FLOAT_TOLERANCE

            if out_of_triangle:
                break
        except np.linalg.LinAlgError:
            out_of_triangle = True
        if out_of_triangle:
            break

    valid = not out_of_triangle
    return bool(valid), bool(out_of_triangle)


def evaluate(code: str, config: Dict[str, Any]):
    """
    Runs the evolved program, gets the point positions, and calculates the minimum triangle area as the score.
    """
    config = config or {}

    # Initialize the return parameters
    fitness: Dict[str, Any] = {
        "min_triangle_area": 0.0,
    }

    eval_info: Dict[str, Any] = {
        "validity": False,
        "execution_time": 0.0,
    }

    execution_time: float = 0.0
    error_info: Dict[str, Any] = {}

    # Construct the safe execution environment, and inject numpy
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

        if "points" in scope:
            points = np.array(scope["points"], dtype=float)
        elif callable(scope.get("construct_points")):
            points = scope["construct_points"]()
        else:
            raise ValueError("Code did not produce 'points' variable")

        # Check the validity of the return value
        if np.isnan(points).any():
            raise ValueError("NaN values detected in points")

        # Check the shape of the return value
        if points.shape != (11, 2):
            raise ValueError(f"Invalid points shape: {points.shape}, expected (11, 2)")

        # Check the geometric constraints
        valid, out_of_triangle = validate_points(points)

        raw_min_triangle_area = get_min_triangle_area(points)
        min_triangle_area = float(raw_min_triangle_area if valid else 0.0)

        fitness["min_triangle_area"] = min_triangle_area

        # ---------- Record the evaluation information ----------
        eval_info["validity"] = bool(valid)
        if valid:
            eval_info["execution_time"] = execution_time
        else:
            eval_info["execution_time"] = execution_time

            error_info = {
                "type": "ConstraintViolation",
                "message": "Points are outside the predefined equilateral triangle.",
                "details": {
                    "out_of_triangle": out_of_triangle,
                },
            }

    except Exception as e:
        # ---------- Error handling ----------
        fitness["min_triangle_area"] = 0.0
        
        eval_info = {
            "validity": False,
            "execution_time": 0.0,
        }

        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

    return fitness, eval_info, error_info

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    code = "points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254037844386], [0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]])"
    fitness, eval_info, error_info = evaluate(code, config)
    print(fitness)
    print(eval_info)
    print(error_info)
    