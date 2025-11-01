import inspect


def example_func(a, b, c=10):
    print(f"a={a}, b={b}, c={c}")
    pass


def get_all_param_names(func):
    sig = inspect.signature(func)
    return set(sig.parameters.keys())


def create_partial_func(func, /, **fixed_kwargs):
    """
    与えられた関数`func`の一部のパラメータを固定値でバインドした新しい関数を動的に作成して返す。
    """
    original_param_names = get_all_param_names(func)

    # 与えられたfixed_kwargsと共通のもの、異なるものを分ける
    fixed_param_names = set(fixed_kwargs.keys())
    original_param_names = get_all_param_names(func)

    # 固定値にするパラメータ名のみを使用するために、共通項目を抽出
    valid_fixed_param_names = fixed_param_names.intersection(original_param_names)
    # 上書きされないパラメータ名を抽出
    remaining_param_names = original_param_names - valid_fixed_param_names

    # 動的に関数を作成
    func_code = f"""
def partial_func({", ".join(remaining_param_names)}):
    combined_kwargs = {{**fixed_kwargs, {", ".join([f"'{name}': {name}" for name in remaining_param_names])}}}
    return func(**combined_kwargs)
"""
    local_scope = {"fixed_kwargs": fixed_kwargs, "func": func}
    exec(func_code, local_scope)
    return local_scope["partial_func"]


if __name__ == "__main__":
    partial = create_partial_func(example_func, a=1, c=20)
    print(inspect.signature(partial))
    partial(2)
