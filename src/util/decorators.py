"""
デコレーター群
"""

from functools import wraps
from typing import Any, Callable, Protocol, TypeVar, cast

R_co = TypeVar("R_co", covariant=True)


class _RunOnceCallable(Protocol[R_co]):
    _called: bool
    _result: object

    def __call__(self, *args: Any, force: bool = False, **kwargs: Any) -> R_co: ...

    def reset(self) -> None: ...


def run_once(func: Callable[..., R_co]) -> _RunOnceCallable[R_co]:
    """
    関数を一度だけ実行し、その結果をキャッシュするデコレーター。
    再度実行したい場合は、force=Trueを引数に渡すか、resetメソッドを呼び出す。

    Example:
        @run_once
        def setup_config():
            print("実際に初期化実行")
            return {"env": "production"}

        setup_config()  # 実行される
        setup_config()  # キャッシュから返る
        setup_config(force=True)  # 強制再実行
        setup_config.reset()  # リセットして次回再実行可能に
    """

    _MISSING = object()

    @wraps(func)
    def wrapper(*args: Any, force: bool = False, **kwargs: Any) -> R_co:
        wrapped = cast(_RunOnceCallable[R_co], wrapper)
        if not wrapped._called or force:
            wrapped._result = func(*args, **kwargs)
            wrapped._called = True

        if wrapped._result is _MISSING:
            raise RuntimeError("run_once internal error: result is missing")
        return cast(R_co, wrapped._result)

    wrapped = cast(_RunOnceCallable[R_co], wrapper)
    wrapped._called = False
    wrapped._result = _MISSING

    def reset() -> None:
        wrapped._called = False
        wrapped._result = _MISSING

    wrapped.reset = reset
    return wrapped
