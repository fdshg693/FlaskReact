# テンプレート管理システム

本プロジェクトでは、テンプレートファイルからGitHub Copilotの設定ファイルを自動生成する仕組みを採用しています。

---

## 📖 概要

### 仕組み

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

### メリット

1. **階層的な管理**: カテゴリ・タイプでフォルダ分けして整理可能
2. **フラットなファイル名**: GitHub Copilotの制約（サブフォルダ不可）に対応
3. **変数置換**: 1つのテンプレートから複数バリエーションを生成可能

---

## 📁 ディレクトリ構造

### テンプレート側

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

### 生成先

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

---

## 📝 命名規則

### エージェント

```
テンプレート:
  .github_copilot_template/{category}/{type}/.agent.md

生成後:
  .github/agents/{category}.{type}.{output-name}.agent.md
```

**例**:
```
.github_copilot_template/coder/script/.agent.md
    ↓
.github/agents/coder.script.default.agent.md
```

### プロンプト

```
テンプレート:
  .github_copilot_template/{category}/{type}/{prompt-name}.prompt.md

生成後:
  .github/prompts/{category}.{type}.{prompt-name}.prompt.md
```

**例**:
```
.github_copilot_template/coder/script/refactor.prompt.md
    ↓
.github/prompts/coder.script.refactor.prompt.md
```

### タスク

タスクファイルはエージェントから生成されます：

```
生成元:
  .github/agents/{category}.{type}.{output}.agent.md

生成先:
  .github/tasks/{category}.{type}.md
```

---

## 📝 エージェント定義の書き方

### フロントマター

```yaml
---
description: エージェントの説明
tools: ['edit', 'search', 'runCommands']
outputs:
  - name: default
  - name: debug
    variables:
      log_level: DEBUG
      verbose: true
---
```

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `description` | ✅ | エージェントの説明 |
| `tools` | ❌ | 利用可能なツール |
| `outputs` | ❌ | 出力バリエーション |

### 出力バリエーション（outputs）

`outputs` フィールドで、1つのテンプレートから複数のエージェントを生成できます：

```yaml
outputs:
  - name: default           # coder.script.default.agent.md
  - name: debug             # coder.script.debug.agent.md
    variables:
      log_level: DEBUG
  - name: strict            # coder.script.strict.agent.md
    variables:
      strictness: high
```

### 本文

```markdown
# Role
エージェントの役割を明確に定義

# Constraints
- 制約1
- 制約2

# Workflow
1. ステップ1
2. ステップ2
```

---

## 🔄 変数置換

### 利用可能な変数

| 変数 | 説明 | 例 |
|------|------|---|
| `${custom:variable_name}` | カスタム変数 | `${custom:log_level}` |
| `${category}` | カテゴリ名 | `coder` |
| `${type}` | タイプ名 | `script` |
| `${output}` | 出力名 | `default` |

### 使用例

**テンプレート**:
```markdown
# Settings
- Log Level: ${custom:log_level}
- Category: ${category}
- Type: ${type}
```

**フロントマター**:
```yaml
outputs:
  - name: debug
    variables:
      log_level: DEBUG
```

**生成後**:
```markdown
# Settings
- Log Level: DEBUG
- Category: coder
- Type: script
```

---

## 📝 プロンプト定義の書き方

### フロントマター

```yaml
---
agent: coder.script.default
---
```

### 本文

```markdown
---
agent: coder.script.default
---

read .github/tasks/coder.script.md to understand your task.

{追加の指示}
```

---

## 🔄 開発ワークフロー

```
1. テンプレート編集
   └─ .github_copilot_template/ 配下を編集

2. エージェントのデプロイ
   └─ deploy-agents

3. プロンプト作成（手動 or 自動生成）
   └─ create-default-prompt

4. タスク生成
   └─ create-task-from-agent

5. 動作確認
   └─ VSCodeでエージェント・プロンプトを使用
```

---

## 🐛 トラブルシューティング

### エージェントが生成されない

```bash
# テンプレートファイルの存在確認
find .github_copilot_template -name ".agent.md"

# フロントマターの構文確認（YAMLとして正しいか）
python -c "import yaml; yaml.safe_load(open('.github_copilot_template/coder/script/.agent.md').read().split('---')[1])"
```

### 変数が置換されない

- 変数名が正しいか確認（`${custom:variable_name}`）
- `outputs` フィールドに `variables` が定義されているか確認
- 変数名がスネークケースか確認（ケバブケースは非対応）

---

## 📚 関連ドキュメント

- [デプロイスクリプト](./deploy_scripts.md)
- [GitHub Copilot 活用方針](./README.md)
