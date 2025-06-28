import json


def jsonTo2DemensionalArray(inputData):

    # 1. 出力したい列の順序をキー名のリストとして定義
    desired_keys = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

    # 2. 定義したキーの順序に従って、各辞書から値を取得し2次元配列に変換
    # lamda関数を別関数に代入させる。
    two_dimensional_array = list(
        map(lambda item: [item[key] for key in desired_keys], inputData)
    )

    return two_dimensional_array


if __name__ == "__main__":
    inputData = [
        {
            "sepal.length": "5.1",
            "sepal.width": "3.5",
            "petal.length": "1.4",
            "petal.width": ".2",
        },
        {
            "petal.width": ".2",
            "sepal.width": "3",
            "petal.length": "1.4",
            "sepal.length": "4.9",
        },
        {
            "sepal.length": "4.7",
            "sepal.width": "3.2",
            "petal.length": "1.3",
            "petal.width": ".2",
        },
    ]

    print(jsonTo2DemensionalArray(inputData))
