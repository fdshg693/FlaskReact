"""
変数置換に関するユーティリティ関数

${custom:name} 形式の変数を置換する機能を提供します。
"""

import re
from typing import Any


def substitute_custom_variables(
    content: str,
    output_values: dict[str, Any],
    custom_inputs: dict[str, str],
) -> str:
    """
    ${custom:name} 形式の変数を置換する

    置換ルール:
    - output_valuesの値が "default" の場合: ${input:name:"default_value"} に変換
    - それ以外の場合: output_valuesの値で直接置換

    Args:
        content: 置換対象のコンテンツ
        output_values: outputエントリの値（nameを除く）
        custom_inputs: custom_inputsの {name: default} 辞書

    Returns:
        str: 置換後のコンテンツ
    """
    # ${custom:name} パターンを検出
    pattern = r"\$\{custom:([^}]+)\}"

    def replace_match(match: re.Match[str]) -> str:
        var_name = match.group(1)

        # output_valuesに対応する値があるか確認
        if var_name in output_values:
            value = output_values[var_name]

            if value == "default":
                # defaultの場合は ${input:name:"default_value"} 形式に変換
                default_value = custom_inputs.get(var_name, "")
                return f'${{input:{var_name}:"{default_value}"}}'
            else:
                # それ以外は直接値で置換
                return str(value)
        else:
            # output_valuesにない場合はそのまま残す
            return match.group(0)

    return re.sub(pattern, replace_match, content)


def get_output_values_without_name(output_entry: dict[str, Any]) -> dict[str, Any]:
    """
    outputエントリからnameを除いた値を取得する

    Args:
        output_entry: outputエントリ

    Returns:
        dict: nameを除いた値の辞書
    """
    result = output_entry.copy()
    result.pop("name", None)
    return result
