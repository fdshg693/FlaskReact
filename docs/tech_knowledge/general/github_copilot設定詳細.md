# GitHub Copilot 設定ファイル詳細仕様

> **Note**: このドキュメントは技術的な詳細を記載したリファレンスです。
> 基本的な使い方は [../../project_overview/05_AI活用・自動化.md](../../project_overview/05_AI活用・自動化.md) を参照してください。

---

## 📚 目次

1. [GitHub Copilot の標準機能](#1-github-copilot-の標準機能)
2. [本プロジェクトのテンプレート管理システム](#2-本プロジェクトのテンプレート管理システム)
3. [自動生成スクリプト](#3-自動生成スクリプト)
4. [クイックリファレンス](#4-クイックリファレンス)

---

## 1. GitHub Copilot の標準機能

GitHub Copilotは以下の3種類の設定ファイルをサポートしています。

### 1.1 エージェント（Agents）

| 項目 | 内容 |
|------|------|
| **配置場所** | `.github/agents/` 直下（サブフォルダ不可） |
| **ファイル名規則** | `{agent-name}.agent.md` |
| **用途** | 特定の役割を持つエージェントの定義 |
| **呼び出し方** | UI上から選択、または `@agent-name` |

**定義できる内容:**
- エージェント固有のプロンプト（ロール、制約、ワークフロー）
- 利用可能なツール（`tools` フィールド）
- 出力バリエーション（`outputs` フィールド）

**フロントマター例:**
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
```

**本文の構成:**
```markdown
# Role
エージェントの役割を明確に定義

# Constraints
制約事項・守るべきルール

# Workflow
処理の流れ・ステップ
```

### 1.2 プロンプトファイル（Prompts）

| 項目 | 内容 |
|------|------|
| **配置場所** | `.github/prompts/` 直下（サブフォルダ不可） |
| **ファイル名規則** | `{prompt-name}.prompt.md` |
| **用途** | 特定の目的に特化したプロンプトテンプレート |
| **呼び出し方** | スラッシュコマンド（`/prompt-name`） |

**フロントマター例:**
```yaml
---
agent: general.basic.default  # 使用するエージェント名
---
```

**プロンプト内容:**
```markdown
---
agent: coder.script.default
---

このファイルのリファクタリングを実施してください。
以下の観点でチェック：
- PEP 8準拠
- 型ヒントの追加
- docstringの充実化
- 関数の適切な分割
```

### 1.3 Instructionsファイル（本プロジェクトでは不使用）

| 項目 | 内容 |
|------|------|
| **配置場所** | `.github/copilot-instructions.md` または `.github/instructions/` |
| **ファイル名規則** | `{instruction-name}.instruction.md` |
| **用途** | 毎回自動的に適用される指示 |

> ⚠️ **本プロジェクトでは Instructionsファイルを使用しません**
>
> **理由:**
> - 毎回適用されるプロンプトは最小限に抑えるべき
> - コンテキストの肥大化を防ぐ
> - 各エージェント・プロンプトに特化したコンテキストのみを与える

---

## 2. 本プロジェクトのテンプレート管理システム

### 2.1 概要

テンプレートファイルから GitHub Copilot の設定ファイルを自動生成する仕組みを採用しています。

```
.github_copilot_template/       ← テンプレート（編集対象）
    └── {category}/{type}/
            ├── .agent.md       # エージェント定義
            └── *.prompt.md     # プロンプト

        ↓ スクリプトで変換

.github/                        ← 生成されたファイル（自動生成）
    ├── agents/
    │   └── {category}.{type}.{output}.agent.md
    └── prompts/
        └── {category}.{type}.{name}.prompt.md
```

### 2.2 ディレクトリ構造

**テンプレート側（`.github_copilot_template/`）:**
```
.github_copilot_template/
├── coder/                      # カテゴリ: コード生成系
│   └── script/                 # タイプ: スクリプト
│       ├── .agent.md           # エージェント定義
│       ├── default.prompt.md   # デフォルトプロンプト
│       └── refactor.prompt.md  # リファクタリング用プロンプト
├── docs/                       # カテゴリ: ドキュメント系
│   ├── ai_knowledge/
│   │   └── .agent.md
│   ├── readme/
│   │   └── .agent.md
│   └── review/
│       └── .agent.md
└── general/                    # カテゴリ: 汎用
    ├── basic/
    │   └── .agent.md
    └── folder_specific/
        └── .agent.md
```

**生成先（`.github/`）:**
```
.github/
├── agents/
│   ├── coder.script.default.agent.md
│   ├── docs.ai_knowledge.default.agent.md
│   ├── docs.readme.default.agent.md
│   ├── docs.review.default.agent.md
│   ├── general.basic.default.agent.md
│   └── general.folder_specific.default.agent.md
├── prompts/
│   ├── coder.script.default.prompt.md
│   └── coder.script.refactor.prompt.md
└── tasks/
    ├── coder.script.md
    ├── docs.ai_knowledge.md
    └── general.basic.md
```

### 2.3 命名規則

#### エージェント

```
テンプレート:
  .github_copilot_template/{category}/{type}/.agent.md

生成後:
  .github/agents/{category}.{type}.{output-name}.agent.md
```

**例:**
```
.github_copilot_template/coder/script/.agent.md
    ↓
.github/agents/coder.script.default.agent.md
```

#### プロンプト

```
テンプレート:
  .github_copilot_template/{category}/{type}/{prompt-name}.prompt.md

生成後:
  .github/prompts/{category}.{type}.{prompt-name}.prompt.md
```

**例:**
```
.github_copilot_template/coder/script/refactor.prompt.md
    ↓
.github/prompts/coder.script.refactor.prompt.md
```

---

## 3. 自動生成スクリプト

### 3.1 エージェントファイルのデプロイ

**スクリプト:** `scripts/github_copilot/template_handle/deploy_agents.py`

**テストファイル:** `scripts/github_copilot/template_handle/tests/test_deploy_agents.py`

#### 主な機能

1. **パス変換**: ディレクトリ構造を `.` 区切りのファイル名に変換
2. **複数バージョン出力**: `outputs` フィールドで複数のエージェントを生成
3. **変数置換**: `${custom:name}` 形式の変数を置換

#### フロントマター処理

**入力（テンプレート）:**
```yaml
---
description: スクリプト生成エージェント
tools: ['edit', 'search', 'runCommands']
outputs:
  - name: default
  - name: debug
    variables:
      log_level: DEBUG
      verbose: true
---
```

**出力1（default）:**
```markdown
<!-- .github/agents/coder.script.default.agent.md -->
---
description: スクリプト生成エージェント
tools: ['edit', 'search', 'runCommands']
---
[エージェントの本文]
```

**出力2（debug）:**
```markdown
<!-- .github/agents/coder.script.debug.agent.md -->
---
description: スクリプト生成エージェント
tools: ['edit', 'search', 'runCommands']
---
[エージェントの本文（変数置換後）]
- log_level: DEBUG
- verbose: true
```

#### 実行方法

```bash
# 全エージェントをデプロイ
python scripts/github_copilot/template_handle/deploy_agents.py

# 設定ファイル指定
python scripts/github_copilot/template_handle/deploy_agents.py config.yaml

# クリーンデプロイ（既存ファイルを削除後にデプロイ）
python scripts/github_copilot/template_handle/deploy_agents.py --clean

# デバッグモード
python scripts/github_copilot/template_handle/deploy_agents.py --verbose
```

### 3.2 デフォルトプロンプトファイルの生成

**スクリプト:** `scripts/github_copilot/template_handle/create_default_prompt.py`

#### 機能

- `.github_copilot_template/` 配下の最下層ディレクトリを検出
- 各ディレクトリに `default.prompt.md` を生成（既に存在する場合はスキップ）

#### 生成されるファイルの内容

```markdown
---
agent: {category}.{type}.default
---
read .github/tasks/{category}.{type}.md to understand your task.
```

**例:**
```markdown
---
agent: coder.script.default
---
read .github/tasks/coder.script.md to understand your task.
```

#### 実行方法

```bash
python scripts/github_copilot/template_handle/create_default_prompt.py
```

> ⚠️ **注意**: これは簡便のための生成であり、実際の使用時は適宜修正が必要です。

### 3.3 タスクファイルの生成

**スクリプト:** `scripts/github_copilot/template_handle/create_task_from_agent.py`

#### 機能

- `.github/agents/` 配下のエージェント定義を読み込み
- 対応するタスクファイルを `.github/tasks/` に生成
- エージェントの説明、制約、ワークフローを抽出・整形

#### 実行方法

```bash
python scripts/github_copilot/template_handle/create_task_from_agent.py
```

### 3.4 プロンプトファイルのデプロイ（未実装）

**スクリプト:** `scripts/github_copilot/template_handle/deploy_prompts.py`（未実装）

#### 実装予定の仕様

- パス変換: `.github_copilot_template/{category}/{type}/{name}.prompt.md`  
  → `.github/prompts/{category}.{type}.{name}.prompt.md`
- フロントマターの `agent` フィールドで指定されたエージェントが存在するか検証
- 存在しない場合は警告を出力してスキップ

---

## 4. クイックリファレンス

### 4.1 ファイル命名規則まとめ

| 種類 | テンプレート | 生成後 |
|------|-------------|--------|
| **エージェント** | `{category}/{type}/.agent.md` | `.github/agents/{category}.{type}.{output-name}.agent.md` |
| **プロンプト** | `{category}/{type}/{name}.prompt.md` | `.github/prompts/{category}.{type}.{name}.prompt.md` |
| **タスク** | （エージェントから生成） | `.github/tasks/{category}.{type}.md` |

### 4.2 関連ファイル一覧

| ファイル/ディレクトリ | 説明 |
|---------------------|------|
| `.github_copilot_template/` | テンプレートファイル格納場所（編集対象） |
| `.github/agents/` | 生成されたエージェントファイル（自動生成） |
| `.github/prompts/` | プロンプトファイル |
| `.github/tasks/` | タスク定義ファイル（エージェントから生成） |
| `scripts/github_copilot/template_handle/` | 変換スクリプト群 |

### 4.3 開発ワークフロー

```
1. テンプレート編集
   └─ .github_copilot_template/ 配下を編集

2. エージェントのデプロイ
   └─ python scripts/github_copilot/template_handle/deploy_agents.py

3. プロンプト作成（手動 or 自動生成）
   └─ python scripts/github_copilot/template_handle/create_default_prompt.py

4. タスク生成
   └─ python scripts/github_copilot/template_handle/create_task_from_agent.py

5. 動作確認
   └─ VSCodeでエージェント・プロンプトを使用
```

### 4.4 変数置換

エージェント本文内で以下の変数を使用可能：

| 変数 | 説明 | 例 |
|------|------|---|
| `${custom:variable_name}` | カスタム変数 | `${custom:log_level}` |
| `${category}` | カテゴリ名 | `coder` |
| `${type}` | タイプ名 | `script` |
| `${output}` | 出力名 | `default` |

**使用例:**

```markdown
<!-- テンプレート -->
# Settings
- Log Level: ${custom:log_level}
- Category: ${category}
- Type: ${type}
```

```yaml
# フロントマター
outputs:
  - name: debug
    variables:
      log_level: DEBUG
```

**生成後:**
```markdown
# Settings
- Log Level: DEBUG
- Category: coder
- Type: script
```

---

## 📚 関連ドキュメント

- **基本的な使い方**: [../../project_overview/05_AI活用・自動化.md](../../project_overview/05_AI活用・自動化.md)
- **スクリプト活用**: [../../project_overview/04_スクリプト・ツール活用.md](../../project_overview/04_スクリプト・ツール活用.md)
- **プロジェクト構造**: [../../dev_contract/01_プロジェクト構造.md](../../dev_contract/01_プロジェクト構造.md)

---

## 🐛 トラブルシューティング

### エージェントが生成されない

```bash
# テンプレートファイルの存在確認
find .github_copilot_template -name ".agent.md"

# フロントマターの構文確認（YAMLとして正しいか）
python -c "import yaml; yaml.safe_load(open('.github_copilot_template/coder/script/.agent.md').read().split('---')[1])"

# デバッグモードで実行
python scripts/github_copilot/template_handle/deploy_agents.py --verbose
```

### 変数が置換されない

- 変数名が正しいか確認（`${custom:variable_name}`）
- `outputs` フィールドに `variables` が定義されているか確認
- 変数名がスネークケースか確認（ケバブケースは非対応）

### プロンプトが認識されない

- ファイル名が `*.prompt.md` で終わっているか確認
- `.github/prompts/` に配置されているか確認
- フロントマターに `agent` フィールドが存在するか確認
- 指定したエージェントが `.github/agents/` に存在するか確認

---

## 📝 更新履歴

| 日付 | 内容 |
|------|------|
| 2025-11-30 | プロジェクト概要から技術ナレッジへ移動・整理 |
