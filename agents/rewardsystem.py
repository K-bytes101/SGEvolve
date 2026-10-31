import importlib.util
from pathlib import Path
import inspect
import multiprocessing
import os
from typing import Any, Dict, List
from utils import format_obj_list


class TimeoutError(Exception):
    pass

def _load_evaluator_target(config):
    """
    Load evaluator in the evaluation subprocess and return the callable object and its parameter signature information

    Args:
        config: Configuration

    Returns:
        target: Callable object
        takes_config: Whether to need configuration
    """
    evaluator_path = Path(config.get("evaluator_path", ""))
    if not evaluator_path:
        raise ValueError("config['evaluator'] not set: need to specify the .py file path of the problem evaluator.")

    abs_path = evaluator_path.absolute()
    if not abs_path.is_file():
        raise FileNotFoundError(f"Evaluator file not found: {abs_path}")

    spec = importlib.util.spec_from_file_location("dynamic_eval_worker", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {abs_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = None
    if hasattr(module, "evaluate") and callable(getattr(module, "evaluate")):
        target = getattr(module, "evaluate")
    else:
        for cls_name in ("Evaluator", "RewardSystem"):
            if hasattr(module, cls_name):
                cls = getattr(module, cls_name)
                if callable(getattr(cls, "evaluate", None)):
                    target = cls(config)
                    break

    if target is None:
        raise AttributeError("Evaluator function or class not found")

    takes_config = len(inspect.signature(target).parameters) >= 2
    return target, takes_config


def _worker_loop(config, conn):
    """
    Resident evaluation process: loop to receive requests and return evaluation results

    Args:
        config: Configuration
        conn: Pipe
    """
    # Evaluators do not need LLM credentials. Clear them in this process only,
    # before loading evaluator code; this is not a general-purpose OS sandbox.
    for name in tuple(os.environ):
        normalized = name.upper()
        if normalized in ("API_KEY", "API_TOKEN") or normalized.endswith(("_API_KEY", "_API_TOKEN")):
            os.environ.pop(name, None)

    try:
        target, takes_config = _load_evaluator_target(config)
    except Exception as e:
        # Startup phase failed, directly tell the parent process and exit
        conn.send({"type": "boot_error", "exception": str(e)})
        conn.close()
        return

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        msg_type = msg.get("type")
        if msg_type == "shutdown":
            break
        if msg_type != "eval":
            conn.send({"success": False, "exception": f"Unknown message type: {msg_type}"})
            continue

        solution = msg.get("solution", "")
        try:
            raw_result = target(solution, config) if takes_config else target(solution)
            if not (
                isinstance(raw_result, tuple)
                and len(raw_result) == 3
                and isinstance(raw_result[0], dict)
                and isinstance(raw_result[1], dict)
                and isinstance(raw_result[2], dict)
            ):
                raise TypeError(
                    "The underlying evaluator.evaluate should return a fixed format (fitness_dict, detail_info_dict, error_info_dict), "
                    f"the actual return type/structure is: {type(raw_result)}"
                )
            conn.send({"success": True, "raw_result": raw_result})
        except Exception as e:
            conn.send({"success": False, "exception": str(e)})

    conn.close()

class RewardSystem:
    """
    Generic evaluator adapter

    Responsibilities:
    - Load the problem-specific evaluator module from the given path
    - Call the input evaluate function
    - Parse the return result, get multi-objective fitness: List[float]

    Agreement:
    - config["evaluator"] specifies the evaluator file path (relative or absolute, pointing to *.py)
    - The module must provide an evaluate function:
        def evaluate(solution: str) -> Any
      or provide a class with an evaluate method: Evaluator/RewardSystem:
        class Evaluator:
            def __init__(self, config: dict): ...
            def evaluate(self, solution: str) -> Any

    - evaluate:
        return: fitness: Dict[str, Any]
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.evaluator_path = self.config.get("evaluator_path")
        evaluator_path_obj = Path(self.evaluator_path) if self.evaluator_path else None
        self._worker_process = None
        self._parent_conn = None

        self.objective_list = format_obj_list(self.config['prompt'].get('objectives', []))
        if not self.objective_list:
            raise ValueError("Objectives are not found, please check the objectives configuration.")

        if not evaluator_path_obj or not evaluator_path_obj.exists():
            raise FileNotFoundError(f"Invalid evaluator path: {self.evaluator_path}")

    def _start_worker(self):
        if self._worker_process is not None and self._worker_process.is_alive():
            return

        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=_worker_loop,
            args=(self.config, child_conn),
            daemon=True,
        )
        p.start()
        child_conn.close()

        self._worker_process = p
        self._parent_conn = parent_conn

        # Quick check for startup phase errors (if the evaluator fails to load, the subprocess will immediately send boot_error)
        if parent_conn.poll(0.5):
            msg = parent_conn.recv()
            if msg.get("type") == "boot_error":
                self._stop_worker()
                raise RuntimeError(f"Evaluation process startup failed: {msg.get('exception')}")

    def _stop_worker(self):
        if self._parent_conn is not None:
            try:
                self._parent_conn.send({"type": "shutdown"})
            except Exception:
                pass
            try:
                self._parent_conn.close()
            except Exception:
                pass
            self._parent_conn = None

        if self._worker_process is not None:
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)
            self._worker_process = None

    def _restart_worker(self):
        self._stop_worker()
        self._start_worker()

    # ---------- Dynamic loading of the underlying evaluator ----------

    def evaluate(self, solution: str) -> tuple[List[float], Dict[str, Any], Dict[str, Any]]:
        """
        Call the evaluate method of the underlying evaluator, parse and return the multi-objective fitness dictionary

        Args:
            solution: The solution to be evaluated
        """
        timeout_seconds = self.config["evaluation"].get("timeout", 60)  # Default timeout is 60 seconds
        self._start_worker()

        # Read the objectives definition from the config, get the specific property information of fitness
        prompt_cfg = self.config.get("prompt") or {}
        objectives = prompt_cfg.get("objectives") or []
        if not objectives:
            raise ValueError("config['prompt']['objectives'] not configured, cannot construct multi-objective fitness list")

        # Send request to the resident evaluation subprocess
        try:
            self._parent_conn.send({"type": "eval", "solution": solution})
        except Exception:
            # Restart once and retry when pipe exception occurs
            self._restart_worker()
            self._parent_conn.send({"type": "eval", "solution": solution})

        if not self._parent_conn.poll(timeout_seconds):
            # Restart evaluation subprocess when timeout, avoid dead state affecting subsequent tasks
            self._restart_worker()
            # Construct timeout information
            fitness = [float("inf") if obj["direction"] == "min" else float("-inf") for obj in self.objective_list]
            eval_info = {
                "validity": False,
                "execution_time": float(timeout_seconds),
            }
            error_info = {
                "type": "TimeoutError",
                "message": f"Evaluation Timeout ({timeout_seconds}s): "
                           f"The execution exceeded the limit. This is typically due to high algorithmic complexity, "
                           f"suboptimal optimization, or logical deadlocks in certain input scenarios.",
            }
            return fitness, eval_info, error_info

        msg = self._parent_conn.recv()
        if not msg.get("success", False):
            raise Exception(msg.get("exception"))

        raw_result = msg["raw_result"]
        fitness_dict = raw_result[0]
        eval_info = raw_result[1]
        error_info = raw_result[2]

        # Extract the corresponding values in the order of objectives
        fitness: List[float] = []
        for idx, obj in enumerate(objectives):
            if not isinstance(obj, dict):
                raise TypeError(
                    f"config['prompt']['objectives'][{idx}] is not a dictionary, actual type is {type(obj)}"
                )

            name = obj.get("name")
            if not name:
                raise ValueError(
                    f"config['prompt']['objectives'][{idx}] missing 'name' field, cannot get value from fitness."
                )

            if name not in raw_result[0]:
                raise KeyError(
                    f"The fitness dictionary returned by the underlying evaluator is missing the value corresponding to the target '{name}'."
                )

            value = fitness_dict[name]
            try:
                fitness.append(float(value))
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"The value {value!r} of the target '{name}' cannot be converted to float."
                ) from e

        return fitness, eval_info, error_info

    def close(self):
        self._stop_worker()

    def __del__(self):
        # Bottom-level resource recovery, avoid residual subprocesses when the interpreter exits
        try:
            self._stop_worker()
        except Exception:
            pass
