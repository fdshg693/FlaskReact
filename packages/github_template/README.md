# github-template

GitHub Copilot テンプレート管理ツール

`.github_copilot_template/` 配下のテンプレートファイルを `.github/agents/` や `.github/prompts/` に展開するためのCLIツールです。

## インストール

```bash
# 開発モードでインストール
uv pip install -e packages/github_template

# または pip を使用
pip install -e packages/github_template
```

## 使用方法

インストール後、以下のコマンドが使用可能になります。

### deploy-agents

`.github_copilot_template/` 配下の `.agent.md` ファイルを `.github/agents/` へ展開します。

```bash
# 全エージェントをデプロイ
deploy-agents

# 設定ファイルで指定したエージェントのみデプロイ
deploy-agents deploy_agents.yaml

# クリーンデプロイ（既存ファイルを削除後にデプロイ）
deploy-agents --clean

# 既存ファイルをスキップ
deploy-agents --no-overwrite
```

### deploy-prompts

`.github_copilot_template/` 配下の `.prompt.md` ファイルを `.github/prompts/` へ展開します。

```bash
# 全プロンプトをデプロイ
deploy-prompts

# クリーンデプロイ
deploy-prompts --clean

# エージェント検証をスキップ
deploy-prompts --skip-validation
```

### create-default-prompt

`.github_copilot_template/` 配下の最下層ディレクトリに `default.prompt.md` を自動生成します。

```bash
create-default-prompt
```

### create-task-from-agent

`.github/agents/` 配下の `.agent.md` から `.github/tasks/` にタスクファイルを自動生成します。

```bash
create-task-from-agent
```

## 設定ファイル

### deploy_agents.yaml

デプロイ対象のエージェントをフィルタリングするための設定ファイルです。
プロジェクトルートに配置してください。

```yaml
include:
  - coder/                    # coderディレクトリ配下のすべてのエージェント
  - general/basic             # 特定のエージェントのみ
```

## ディレクトリ構造

```
.github_copilot_template/     # テンプレートソース
├── coder/
│   └── script/
│       └── .agent.md
├── general/
│   └── basic/
│       ├── .agent.md
│       └── default.prompt.md
└── ...

.github/                      # 展開先
├── agents/                   # deploy-agents の出力先
│   ├── coder.script.default.agent.md
│   └── general.basic.default.agent.md
├── prompts/                  # deploy-prompts の出力先
│   └── general.basic.default.prompt.md
└── tasks/                    # create-task-from-agent の出力先
    └── general.basic.md
```

## 開発

```bash
# テストの実行
cd packages/github_template
pytest
```

## ライセンス

MIT
