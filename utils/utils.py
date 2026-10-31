from typing import Dict, Any, List


# --------------------------------- Format optimization objective list ---------------------------------
def format_obj_list(objectives: List[Dict]) -> List[Dict]:
    """
    Format optimization objective list
    """
    obj_list = []
    for idx, obj in enumerate(objectives):
        obj_name = obj.get("name") or obj.get("objective")
        if not obj_name:
            print(f"[Warning] Objective name {idx} not found, using default value")
            obj_name = f"Objective {idx}"

        opt_direction = obj.get("direction") or obj.get("type")
        if not opt_direction:
            raise ValueError(f"Optimization direction is not found for objective: {obj_name}")
        if opt_direction not in ['min', 'max']:
            raise ValueError(f"Optimization direction is invalid for objective: {obj_name}")

        description = obj.get("description")
        if not description:
            raise ValueError(f"Optimization objective description is not found for objective: {obj_name}")

        benchmark = obj.get("benchmark")

        obj_list.append({
            "name": obj_name,
            "direction": opt_direction,
            "description": description,
            "benchmark": benchmark
        })

    return obj_list


# --------------------------------- Replace placeholders ---------------------------------
def replace_placeholders(template: str, replacements: Dict) -> str:
    """
    Replace placeholders in template

    Args:
        template: Template string containing placeholders
        replacements: Dictionary of placeholder names to replacement values

    Returns:
        str: String with placeholders replaced
    """
    result = template

    # Iterate over all placeholders to replace
    for placeholder, value in replacements.items():
        # Format placeholder as {PLACEHOLDER}
        placeholder_pattern = f"{{{placeholder}}}"

        # Ensure replacement value is a string type
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)

        # Execute replacement
        result = result.replace(placeholder_pattern, value)

    return result


# --------------------------------- 安全执行环境实现 ---------------------------------
_DANGEROUS_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "telnetlib",
    "pickle",
    "shelve",
    "dbm",
    "sqlite3",
    "ctypes",
    "threading",
    "multiprocessing",
    "queue",
    "concurrent",
    "asyncio",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Safe __import__ implementation (used by this evaluator).

    - Only do blacklist check: base name in _DANGEROUS_MODULES will be rejected;
    - Other modules will use default __import__.
    """

    base_name = name.split(".")[0]
    if base_name in _DANGEROUS_MODULES:
        raise ImportError(
            f"Import of '{name}' is blocked for security reasons. "
            f"Module '{base_name}' is in the dangerous-modules blacklist."
        )

    return __import__(name, globals, locals, fromlist, level)


def create_safe_execution_environment() -> Dict[str, Any]:
    """Create a safe global execution environment for exec/compile.

    - Replace __import__ in __builtins__ with _safe_import: 
    - Other commonly used built-in functions are preserved;
    - No additional whitelist restrictions, mainly relying on blacklist.
    """

    builtins_safe: Dict[str, Any] = {
        # ---- Common exception types (allow user code to raise / except) ----
        # Note: These exception types do not provide I/O / system capabilities, only for error expression and control flow.
        "BaseException": BaseException,
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AssertionError": AssertionError,
        "ArithmeticError": ArithmeticError,
        "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError,
        "NameError": NameError,

        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__build_class__": __build_class__,
        "__import__": _safe_import,
    }

    return {
        "__builtins__": builtins_safe,
        "__name__": "__main__",
    }


# --------------------------------- Format error message ---------------------------------
def _extract_line_number_from_traceback(code_id: str) -> int | None:
    """
    Extract line number from the most recent traceback if it references the executed code.

    Args:
        code_id: The code identifier used in compile() (e.g., '<code_item_0>')

    Returns:
        Line number (1-indexed) if found, None otherwise
    """
    import sys
    tb = sys.exc_info()[2]
    if tb is None:
        return None

    # Walk through the traceback to find the first frame that references our code
    current_tb = tb
    while current_tb is not None:
        frame = current_tb.tb_frame
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Check if this frame is from our executed code
        if code_id in filename:
            return lineno

        current_tb = current_tb.tb_next

    return None


def get_code_context(code: str, line_num: int, context_lines: int = 2) -> str:
    """
    Get code context around a specific line number for error messages.

    Args:
        code: Code string
        line_num: Line number (1-indexed)
        context_lines: Number of lines to show before and after

    Returns:
        Formatted code context string
    """
    lines = code.split('\n')
    start = max(0, line_num - context_lines - 1)
    end = min(len(lines), line_num + context_lines)

    context = []
    for i in range(start, end):
        line_prefix = ">>> " if i == line_num - 1 else "    "
        context.append(f"{line_prefix}{i+1:4d} | {lines[i]}")

    return '\n'.join(context)


def enhance_error_message(error: Exception, code: str, code_id: str) -> str:
    """
    Enhance error message with code context if the error originates from executed code.

    Args:
        error: The exception object
        code: The executed code string
        code_id: The code identifier used in compile()

    Returns:
        Enhanced error message with code context
    """
    error_msg = str(error)

    # Try to extract line number from traceback
    line_num = _extract_line_number_from_traceback(code_id)

    if line_num is not None:
        # Error originates from executed code, add context
        context = get_code_context(code, line_num)
        enhanced_msg = (
            f"{error_msg}\n"
            f"Error at line {line_num} in executed code:\n"
            f"Code snippet around line {line_num}:\n"
            f"{context}"
        )
        return enhanced_msg

    # If we can't find line number, return original message
    return error_msg