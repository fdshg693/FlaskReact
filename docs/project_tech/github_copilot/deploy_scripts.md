# デプロイスクリプト詳細仕様

テンプレートからGitHub Copilot設定ファイルを生成するスクリプトの詳細仕様です。

---

## 📖 概要

本プロジェクトでは、以下のデプロイスクリプトを提供しています：

| コマンド | 機能 |
|---------|------|
| `deploy-agents` | エージェント定義のデプロイ |
| `create-default-prompt` | デフォルトプロンプトの生成 |
| `deploy-prompts` | プロンプトのデプロイ |
| `create-task-from-agent` | タスクファイルの生成 |

> **注意**: これらのコマンドは `packages/github_template` パッケージで提供されており、`uv sync` で自動的にインストールされます。

---

## 🚀 deploy-agents

エージェント定義をデプロイします。

### 基本使用法

```bash
# すべてのエージェントをデプロイ
deploy-agents

# 設定ファイルを指定してデプロイ
deploy-agents deploy_agents.yaml
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--clean` | 既存ファイルを削除してからデプロイ |
| `--no-overwrite` | 既存ファイルをスキップ |
| `--verbose` | 詳細ログを出力 |

### 処理内容

1. `.github_copilot_template/` のテンプレートを読み込み
2. フロントマターを解析（YAML）
3. `outputs` フィールドで指定されたバリエーションを生成
4. 変数置換を実行
5. `.github/agents/` に出力

### 入出力例

**入力（テンプレート）**:
```
.github_copilot_template/coder/script/.agent.md
```

```yaml
---
description: スクリプト生成エージェント
tools: ['edit', 'search', 'runCommands']
outputs:
  - name: default
  - name: debug
    variables:
      log_level: DEBUG
---

# Role
スクリプト生成を担当

# Log Level
${custom:log_level}
```

**出力1（default）**:
```
.github/agents/coder.script.default.agent.md
```

```yaml
---
description: スクリプト生成エージェント
tools: ['edit', 'search', 'runCommands']
---

# Role
スクリプト生成を担当

# Log Level
${custom:log_level}
```

**出力2（debug）**:
```
.github/agents/coder.script.debug.agent.md
```

```yaml
---
description: スクリプト生成エージェント
tools: ['edit', 'search', 'runCommands']
---

# Role
スクリプト生成を担当

# Log Level
DEBUG
```

---

## 🚀 create-default-prompt

各エージェントに対応するデフォルトプロンプトを生成します。

### 基本使用法

```bash
create-default-prompt
```

### 処理内容

1. `.github_copilot_template/` 配下の最下層ディレクトリを検出
2. 各ディレクトリに `default.prompt.md` を生成
3. 既に存在する場合はスキップ

### 生成されるファイル

```markdown
---
agent: {category}.{type}.default
---
read .github/tasks/{category}.{type}.md to understand your task.
```

**例**:
```markdown
---
agent: coder.script.default
---
read .github/tasks/coder.script.md to understand your task.
```

> **注意**: これは雛形生成であり、実際の使用時は内容を適切に編集してください。

---

## 🚀 deploy-prompts

プロンプトファイルをデプロイします。

### 基本使用法

```bash
# 全プロンプトをデプロイ
deploy-prompts

# クリーンデプロイ
deploy-prompts --clean
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--clean` | 既存ファイルを削除してからデプロイ |
| `--skip-validation` | エージェント検証をスキップ |

### 処理内容

1. `.github_copilot_template/` 配下の `.prompt.md` ファイルを検出
2. パス変換: `{category}/{type}/{name}.prompt.md` → `{category}.{type}.{name}.prompt.md`
3. `.github/prompts/` に出力
4. （デフォルト）フロントマターの `agent` フィールドで指定されたエージェントの存在を検証

---

## 🚀 create-task-from-agent

エージェント定義からタスクファイルを生成します。

### 基本使用法

```bash
create-task-from-agent
```

### 処理内容

1. `.github/agents/` 配下のエージェント定義を読み込み
2. 各エージェントの説明、制約、ワークフローを抽出
3. `.github/tasks/{category}.{type}.md` を生成

### 生成されるファイル

エージェントの内容を整形したタスク定義ファイル：

```markdown
# Task: {category}.{type}

## Description
{エージェントのdescription}

## Constraints
{エージェントのConstraintsセクション}

## Workflow
{エージェントのWorkflowセクション}
```

---

## 🔧 設定ファイル

`deploy_agents.yaml` でデプロイ設定をカスタマイズできます。

### 設定例

```yaml
# deploy_agents.yaml
template_dir: .github_copilot_template
output_dir: .github/agents
clean_before_deploy: false
verbose: false
```

### 設定項目

| 項目 | デフォルト | 説明 |
|------|-----------|------|
| `template_dir` | `.github_copilot_template` | テンプレートディレクトリ |
| `output_dir` | `.github/agents` | 出力ディレクトリ |
| `clean_before_deploy` | `false` | デプロイ前に既存ファイルを削除 |
| `verbose` | `false` | 詳細ログを出力 |

---

## 🐛 トラブルシューティング

### コマンドが見つからない

```bash
# パッケージが正しくインストールされているか確認
uv pip list | grep github-template

# 再インストール
uv sync
```

### エージェントが生成されない

```bash
# デバッグモードで実行
deploy-agents --verbose

# テンプレートファイルの存在確認
find .github_copilot_template -name ".agent.md"
```

### フロントマターのエラー

```bash
# YAMLの構文チェック
python -c "import yaml; yaml.safe_load(open('path/to/.agent.md').read().split('---')[1])"
```

### プロンプトが認識されない

- ファイル名が `*.prompt.md` で終わっているか確認
- `.github/prompts/` に配置されているか確認
- フロントマターに `agent` フィールドが存在するか確認
- 指定したエージェントが `.github/agents/` に存在するか確認

---

## 📚 関連ドキュメント

- [テンプレート管理システム](./template_system.md)
- [GitHub Copilot 活用方針](./README.md)
