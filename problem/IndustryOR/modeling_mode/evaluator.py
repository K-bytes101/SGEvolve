"""
IndustryOR modeling_mode: execute Python + gurobipy code and score by absolute error to ground truth.
Fitness key ``answer_gap`` is minimized (0 = exact match within tolerance is still reported as gap magnitude).
"""
from __future__ import annotations

import io
import contextlib
import math
import re
import time
import traceback
from typing import Any, Dict, Tuple

import gurobipy as gp

from utils import create_safe_execution_environment, get_code_context, enhance_error_message

_FINAL_ANSWER_RE = re.compile(
    r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
    re.DOTALL | re.IGNORECASE,
)

_TOL = 1e-4


def evaluate(code: str, config: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = config or {}

    fitness: Dict[str, Any] = {
        "answer_gap": float("inf")
    }
    eval_info: Dict[str, Any] = {
        "reported_value": None,
        "execution_time": 0.0,
    }
    error_info: Dict[str, Any] = {}

    if "answer" not in config:
        print("[Error] config['answer'] is required for IndustryOR modeling_mode.")
        exit(1)

    ground_truth = float(config["answer"])

    safe_globals = create_safe_execution_environment()
    safe_globals["gp"] = gp

    try:
        stdout_buf = io.StringIO()
        try:
            compiled_code = compile(code, "<industry_or_modeling>", "exec")
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
            with contextlib.redirect_stdout(stdout_buf):
                exec(compiled_code, scope, scope)
        except Exception as exec_error:
            enhanced_msg = enhance_error_message(exec_error, code, "<industry_or_modeling>")
            raise type(exec_error)(enhanced_msg) from exec_error

        execution_time = time.time() - start_time
        captured = stdout_buf.getvalue()
        answer_val: float | None = None

        matches = _FINAL_ANSWER_RE.findall(captured)
        if matches:
            raw = matches[-1].strip()
            try:
                answer_val = float(raw)
            except ValueError:
                raise ValueError(f"FINAL_ANSWER content is not a valid float: {raw!r}")
        else:
            answer_val = scope.get("answer")

        if answer_val is None:
            raise ValueError("No answer found: print <FINAL_ANSWER>...</FINAL_ANSWER> to stdout.")

        eval_info["reported_value"] = answer_val
        eval_info["execution_time"] = execution_time

        if not math.isnan(answer_val) and abs(answer_val - ground_truth) < _TOL:
            fitness["answer_gap"] = 0.0
        else:
            fitness["answer_gap"] = abs(answer_val - ground_truth)
            eval_info["message"] = (
                "Non-zero gap to ground truth. Check your formulation."
            )

    except Exception as e:
        fitness["answer_gap"] = float("inf")
        eval_info["reported_value"] = None
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

    return fitness, eval_info, error_info
