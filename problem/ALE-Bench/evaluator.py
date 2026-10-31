import numpy as np
import ale_bench
from ale_bench.result import JudgeResult
from ale_bench.error import AleBenchError
from typing import Tuple, Dict, Any


def evaluate(code: str, config: Dict[str, Any]) -> Tuple:
    """Evaluate ALE-Bench code."""
    session = config["session"]
    coding_language = config["coding_language"]

    # 预设变量
    fitness: Dict[str, Any] = {
        "score": 0.0,
    }
    eval_info: Dict[str, Any] = {
        "validity": False,
        "average_case_execution_time": 0.0,
    }
    error_info: Dict[str, Any] = {}

    try:
        num_public_cases = 50
        cases = session.case_gen(list(range(num_public_cases)))

        # public eval
        public_result = session.case_eval(
            cases, code, code_language="cpp20", skip_local_visualization=True
        )
        public_case_results = public_result.case_results
        

        if public_result.overall_judge_result == JudgeResult.ACCEPTED:
            average_case_execution_time = float(np.mean([case.execution_time for case in public_case_results]))  # execution time in seconds

            eval_info["validity"] = True
            eval_info["average_case_execution_time"] = average_case_execution_time

            fitness["score"] = public_result.overall_absolute_score / num_public_cases
        else:
            eval_info["validity"] = False
            eval_info["average_case_execution_time"] = 0.0
            error_info["cases_eval_log"] = []
            fitness["score"] = 0.0

            seen_failure_keys: set = set()
            for idx, case in enumerate(public_case_results):
                if case.judge_result != JudgeResult.ACCEPTED:
                    key = (case.message, case.error_str)
                    if key in seen_failure_keys:
                        continue
                    seen_failure_keys.add(key)
                    error_info["cases_eval_log"].append(
                        f"Case {idx+1} failed, message: {case.message}, error: {case.error_str}."
                    )

    except AleBenchError as e:
        eval_info["validity"] = False
        eval_info["average_case_execution_time"] = 0.0
        error_info = {
            "type": "AleBenchError",
            "message": str(e),
        }
    except Exception as e:
        eval_info["validity"] = False
        eval_info["average_case_execution_time"] = 0.0
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
        }

    return fitness, eval_info, error_info