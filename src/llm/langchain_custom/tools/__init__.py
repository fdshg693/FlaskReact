"""
LLM Tools Package

This package provides various tools for LLM applications including:
- Search tools (local document search, Tavily search)
- Core utilities (partial function creation, tool schema creation)
- Other tools (sample tools, PDF, text splitter, image)

利点：
- サブモジュールを意識せずに、toolsパッケージから全ての公開APIをインポート可能
- 新しいツールを追加した際に、__init__.pyを手動で更新する必要がない
欠点：
- import時に全てのサブモジュールを読み込むため、初期化コストが高い
- toolsでラッピングされた関数のインポートは出来ていないようなので、要修正
- IDEで定義元にジャンプできない
"""

from pathlib import Path
import importlib
import inspect

__all__: list[str] = []


# toolsでラッピングされた関数のインポートは出来ていないようなので、要修正
def _import_submodules(package_path: Path, package_name: str) -> None:
    """
    Recursively import all submodules and collect public APIs.

    Args:
        package_path: Path to the package directory
        package_name: Fully qualified package name
    """
    global __all__

    # Iterate through all Python files in the package
    for py_file in package_path.rglob("*.py"):
        # Skip __init__.py files and private modules
        if py_file.name == "__init__.py" or py_file.name.startswith("_"):
            continue

        # Calculate relative module path from package directory
        relative_path = py_file.relative_to(package_path)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        # Build full module name including package
        module_name = f"{package_name}.{'.'.join(module_parts)}"

        # examplesモジュールは除外
        if "examples" in relative_path.stem:
            continue

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Collect public members (not starting with _)
            for name, obj in inspect.getmembers(module):
                if not name.startswith("_"):
                    # Check if it's defined in this module (not imported from elsewhere)
                    if hasattr(obj, "__module__") and obj.__module__ == module_name:
                        # Add to globals and __all__
                        globals()[name] = obj
                        if name not in __all__:
                            __all__.append(name)
        except Exception as e:
            # Log import errors but don't fail
            print(f"Warning: Failed to import {module_name}: {e}")


# Get the current package path
_package_path = Path(__file__).parent

# インポートする側から見たパッケージ名を取得（そのため、インポート側によって変わりうる）
_package_name = __name__

# Auto-import all submodules
_import_submodules(_package_path, _package_name)

# Sort __all__ for consistency
__all__.sort()
