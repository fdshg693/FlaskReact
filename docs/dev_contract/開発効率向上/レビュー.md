# コード品質を定期的に確認するための、レビューの使い方
現状、複数の方法でAIレビューを実施できる。  
個々で判断して、適切な方法を選択すること。  

## 1. Github Copilotを使ったレビュー
- Github Copilotを使って、コードレビューを実施する。
    - ```docs/dev_contract/GithubCopilotの使い方/各ファイル説明.md```を参考にして、レビューを生成する。

## 2. Github Copilot CLIを使ったレビュー
- ```scripts/github_copilot/scripts.sh```に、YAMLファイルの内容を元に、複数フォルダに対して並列でAIエージェントに指示を与えて実行するスクリプトを用意しています。
    - ドキュメント作成・レビュー・調査など並列で実行しても問題ない処理を、コンテキストを限定することで精度を上げながら更に自動化・高速化できます。
    - YAMLファイルの書き方は、```scripts/github_copilot/settings/review.yml.example```を参照してください。

## 3. PRベースのレビュー
- Github Actionを使って、PRベースでAIレビューを実施する方法もあります。
    - ```.github/workflows/``を参照してください。
        - 現状は、単にPRの差分をAIにそのまま渡しているだけで、複雑な分析を行えていません。
- スクリプトを実行して、ローカル環境から、AIレビューを実施することも可能です。
    - ```scripts/ai_review/```を参照してください。
        - 現状は、単にPRの差分をAIにそのまま渡しているだけで、複雑な分析を行えていません。
        - Shell Script, Power Shell, Pythonで提供していく予定です。