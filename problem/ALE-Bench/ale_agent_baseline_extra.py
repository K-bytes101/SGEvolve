"""ALE Agent baseline 结果辅助：读取 gemini baseline JSON。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# 默认与仓库内 baseline 文件路径一致
DEFAULT_GEMINI_BASELINE_JSON = Path(
    "/home/fengkewei/hybridevo/problem/ALE-Bench/ALE_Agent_baseline/"
    "gemini-3-flash-preview-high.json"
)


def load_gemini_baseline_json(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    读取 gemini-3-flash-preview-high baseline JSON。

    文件为单行大 JSON, 结构大致为: 
        {
            "model_name": str,
            "codes": { "<run_id>": { "<problem_id>": "<source_code>", ... }, ... }
        }

    Args:
        path: JSON 文件路径；默认使用 ``DEFAULT_GEMINI_BASELINE_JSON``。

    Returns:
        解析后的字典（顶层含 ``model_name``、``codes`` 等键）。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: 内容不是合法 JSON。
    """
    resolved = path if path is not None else DEFAULT_GEMINI_BASELINE_JSON
    resolved = Path(resolved).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"baseline JSON 不存在: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    return data


if __name__ == "__main__":
    payload = load_gemini_baseline_json()
