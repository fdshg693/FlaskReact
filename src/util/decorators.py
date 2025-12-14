"""
デコレーター群
"""

from functools import wraps


def run_once(func):
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

    @wraps(func)
    def wrapper(*args, force=False, **kwargs):
        if not wrapper._called or force:
            wrapper._result = func(*args, **kwargs)
            wrapper._called = True
        return wrapper._result

    wrapper._called = False
    wrapper._result = None
    wrapper.reset = lambda: setattr(wrapper, "_called", False)
    return wrapper
