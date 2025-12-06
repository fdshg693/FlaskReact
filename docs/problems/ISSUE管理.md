# ISSUEの管理方法
このプロジェクトでは、githubのIssue機能を使ってバグ報告や機能要望を管理しています。Issueの管理には以下の手順とルールを採用しています。

## ISSUEテンプレート
`.github/ISSUE_TEMPLATE/basic_template.md`をベースとして、必要に応じてカスタマイズしたテンプレートを使用します。これにより、報告内容の一貫性と必要情報の網羅性を確保します。

## ラベル管理
`docs/problems/label.yaml`にラベルの定義をまとめています。新しいラベルを追加する場合や既存のラベルを変更する場合は、このYAMLファイルを更新し、以下のコマンドでリポジトリのラベルを同期します。

```bash
# 同期させる場合
uv run python scripts/gh/labels.py sync
# ラベルの同期が取れているかの確認を行う場合
uv run python scripts/gh/labels.py check
```

## Github Copilot Agentの活用
### エージェント設定場所
- テンプレート
    - `.github_copilot_template/github/issue/.agent.md`
- 実際のエージェントファイル
    - `.github/agents/github.issue.default.agent.md`
### エージェントの役割
Github Issue・ラベル管理に特化したエージェントを設定しています。このエージェントは以下の機能を持ちます。
- Github MCPサーバーを通じた操作
- gh CLIコマンドを通じた操作
### 使い方
Agentの選択欄にてこのエージェントを選び、ISSUEの管理やラベル付けを実行させることが可能です。