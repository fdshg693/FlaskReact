## このdataフォルダに格納するファイルのサンプル例

- `ai_agent`フォルダ
    - AIエージェントが`llm/document_search_tools.py`を使ってドキュメント検索を行うときに参照するデフォルトのフォルダ
    - テキストファイルを格納する
        - 1行目にファイルの内容を記載すること。この1行目の内容をもとに、AIエージェントはドキュメントの全文を取得するかどうかを判断する 

- pdfファイル
    - `llm/pdf.py`を使って、文字起こしすることが可能
    - 文字起こしされたファイルは、`llm/pdf.py`の`__main__`の部分を適当に変更して実行することで、`data/extracted_{元のファイル名}_{タイムスタンプ}.txt`に保存される
- pngファイル
    - `llm/image.py`を使って、画像を生成AIに描写させることが可能
    - 画像の生成は、`llm/image.py`の`analyze_image`関数を使う
    - 画像の描写は、`llm/image.py`の`__main__`の部分を適当に変更して実行することで、ターミナルに表示される
- csvファイル
    - 機械学習等に利用するCSVファイル