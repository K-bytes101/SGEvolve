import time
import traceback
import numpy as np
from typing import Dict, Any, Tuple
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from utils import create_safe_execution_environment, get_code_context, enhance_error_message
# 多边框走廊
corridor_poly = Polygon([
    (-5, 0), (0, 0), (0, 5), (1, 5), (1, -1), (-5, -1)
])

def get_sofa_area(get_sofa_pose, steps: int = 500) -> Tuple[dict, float]:
    """
    Calculate the maximum area of the sofa that can pass through the corridor based on the get_sofa_pose function.
    Args:
        get_sofa_pose: The function to get the sofa pose
        steps: The number of sampling steps
    Returns:
        valid: Whether is valid
        sofa_area: The area of the sofa
    """
    valid_info = {
        "valid": True,
        "fail_type": "",
        "message": "",
    }
    intersected_sofa = None

    for theta in np.linspace(0, np.pi / 2, steps):
        x, y = get_sofa_pose(theta)

        # Inverse transformation: transform the corridor into the sofa's reference frame, from the sofa's perspective, the corridor is moving and rotating in the opposite direction
        # Transformation order: first translate (-x, -y), then rotate around (0, 0) by -theta
        p = translate(corridor_poly, -x, -y)
        p = rotate(p, -theta, origin=(0, 0), use_radians=True)
        
        # Calculate the intersection of the corridor shape at all times
        if intersected_sofa is None:
            intersected_sofa = p
        else:
            intersected_sofa = intersected_sofa.intersection(p)
            
        # If the intersection is empty, it means that the path cannot pass through any shape
        if intersected_sofa.is_empty:
            valid_info["valid"] = False
            valid_info["fail_type"] = "Geometric Failure"
            valid_info["message"] = f"Sofa disappeared at theta={theta:.2f} rad. Try adjusting position at this angle."
            break

    return valid_info, intersected_sofa.area

def evaluate(code: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate function: calculate the maximum sofa area that can pass through the given path
    """

    # Initialize the return value
    fitness: Dict[str, Any] = {
        "sofa_area": 0.0,
    }

    eval_info: Dict[str, Any] = {
        "validity": False,
        "execution_time": 0.0,
    }

    error_info: Dict[str, Any] = {}

    # Some preset variables
    execution_time: float = 0.0

    safe_globals = create_safe_execution_environment()
    safe_globals["np"] = np

    # Main test process
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

        if callable(scope.get("get_sofa_pose")):
            get_sofa_pose = scope["get_sofa_pose"]

        else:
            raise ValueError("Code did not produce 'get_sofa_pose' function")

        # Calculate the area of the sofa
        valid_info, sofa_area = get_sofa_area(get_sofa_pose)
        valid = valid_info["valid"]

        # Record the evaluation information
        fitness["sofa_area"] = float(sofa_area)
        eval_info["validity"] = bool(valid)
        eval_info["execution_time"] = float(execution_time)

        if not valid:
            error_info = {
                "type": valid_info["fail_type"],
                "message": valid_info["message"],
            }
    except Exception as e:
        fitness["sofa_area"] = 0.0

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
    code = """
from typing import Tuple
import numpy as np
def get_sofa_pose(theta: float) -> Tuple:
    x = 0
    y = np.sin(theta)
    return x, y
    """
    fitness, eval_info, error_info = evaluate(code)
    print(fitness)
    print(eval_info)
    print(error_info)