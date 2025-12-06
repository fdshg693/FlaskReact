# GitHub Copilot 活用方針（本プロジェクト）

本プロジェクトでのGitHub Copilotの活用方針と設定について説明します。

---

## 📖 基本方針

### 使用する機能

| 機能 | 使用 | 備考 |
|------|:----:|------|
| **エージェント** | ✅ | 役割ごとにカスタマイズ |
| **プロンプト** | ✅ | タスクごとにテンプレート化 |
| **タスクファイル** | ✅ | プロジェクト独自の仕組み |
| **Instructions** | ❌ | コンテキスト肥大化防止のため不使用 |

### ディレクトリ構成

```
.github/
├── agents/           # エージェント定義（自動生成）
├── prompts/          # プロンプトテンプレート
└── tasks/            # タスク定義（エージェントから生成）

.github_copilot_template/   # テンプレートファイル（編集対象）
├── coder/
├── docs/
└── general/
```

---

## 🔧 本プロジェクトの特徴

### 1. テンプレート管理システム

本プロジェクトでは、テンプレートファイルからGitHub Copilotの設定ファイルを自動生成する仕組みを採用しています。

**メリット**:
- 階層的なフォルダ構造で管理可能
- 変数置換による複数バージョン生成
- テンプレートの一元管理

詳細: [template_system.md](./template_system.md)

### 2. タスクファイル

プロンプトから参照されるタスク定義ファイルを使用して、詳細な指示を分離しています。

**構造**:
```
プロンプト (.prompt.md)
    └── タスクファイル参照 (.github/tasks/*.md)
        └── 詳細な指示・ワークフロー
```

### 3. デプロイスクリプト

テンプレートから設定ファイルへの変換を自動化するスクリプトを提供しています。

詳細: [deploy_scripts.md](./deploy_scripts.md)

---

## 🚀 利用可能なエージェント

本プロジェクトで定義されているエージェントカテゴリ：

| カテゴリ | 説明 | 例 |
|---------|------|-----|
| `coder.*` | コード生成・編集 | `coder.script.default` |
| `docs.*` | ドキュメント生成 | `docs.readme.default` |
| `general.*` | 汎用 | `general.basic.default` |

エージェント一覧は `.github/agents/` を参照してください。

---

## 📝 基本的な使い方

### エージェントの呼び出し

1. VSCodeでCopilot Chatを開く（`Cmd+Shift+I`）
2. 画面上部のドロップダウンからエージェントを選択
3. 指示を入力

### プロンプトの呼び出し

1. Copilot Chatで `/` を入力
2. 候補からプロンプトを選択
3. 追加の指示があれば入力

### エージェント・プロンプトの追加

1. `.github_copilot_template/` でテンプレートを作成・編集
2. デプロイスクリプトを実行: `deploy-agents`
3. VSCodeでCopilotを再読み込み

---

## 📚 詳細ドキュメント

- [テンプレート管理システム](./template_system.md) - テンプレート構造・命名規則
- [デプロイスクリプト](./deploy_scripts.md) - 各コマンドの詳細仕様

---

## 📚 関連ドキュメント

### 汎用的な技術情報

- [GitHub Copilot 概要](../../tech_knowledge/general/github_copilot/README.md)
- [エージェント機能](../../tech_knowledge/general/github_copilot/agents.md)
- [プロンプト機能](../../tech_knowledge/general/github_copilot/prompts.md)

### プロジェクトドキュメント

- [スクリプト・ツール活用](../../project_overview/04_スクリプト・ツール活用.md)
- [AI活用・自動化](../../project_overview/05_AI活用・自動化.md)
