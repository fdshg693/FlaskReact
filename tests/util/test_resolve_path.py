from pathlib import Path
import textwrap
import pytest

from util.resolve_path import resolve_path


def write_temp_path_env(tmp_dir: Path, content: str) -> Path:
    path_env = tmp_dir / "path.env"
    path_env.write_text(textwrap.dedent(content), encoding="utf-8")
    return path_env


def test_resolve_path_basic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # create isolated project root structure similar to real one
    project_root = tmp_path
    (project_root / "src" / "util").mkdir(parents=True)

    path_env = write_temp_path_env(
        project_root,
        """
    TEXT_DOCUMENTS_PATH=data/llm/text_documents
    # comment line
    IMAGE_PATH = data/llm/image
    """,
    )

    # copy module file into this temp root so that resolve_path uses parents correctly
    original_module = Path(__file__).parents[2] / "src" / "util" / "resolve_path.py"
    temp_module = project_root / "src" / "util" / "resolve_path.py"
    temp_module.write_text(
        original_module.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # ensure directories exist to test existence warning logic (one exists, one missing)
    (project_root / "data" / "llm" / "text_documents").mkdir(parents=True)

    # point PATH_ENV_FILE to our temp path.env
    monkeypatch.setenv("PATH_ENV_FILE", str(path_env))

    # modify sys.path so import util.resolve_path loads temp module
    import sys

    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    from util.resolve_path import resolve_path as temp_resolve

    p1 = temp_resolve("TEXT_DOCUMENTS_PATH")
    assert p1.is_absolute()
    assert p1.name == "text_documents"

    p2 = temp_resolve("IMAGE_PATH")
    assert p2.is_absolute()
    assert p2.name == "image"


def test_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        resolve_path("NOT_DEFINED_KEY")


def test_empty_key_raises() -> None:
    with pytest.raises(KeyError):
        resolve_path("")
