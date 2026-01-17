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
...

```

### 生成先

```
.github/
├── agents/
│   ├── coder.script.default.agent.md
│   ...
│
├── prompts/
│   ├── coder.script.default.prompt.md
│   ├── coder.script.refactor.prompt.md
│   ....
│
└── tasks/
    ├── coder.script.md
    ....
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
      log_level: ERROR
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

---

## 🔄 開発ワークフロー

```
1. テンプレート編集
   └─ .github_copilot_template/ 配下を編集

2. エージェントのデプロイ
   └─ deploy-agents

3. プロンプト作成（手動 or 自動生成）
   └─ create-default-prompt

4. タスク生成(任意)
   └─ create-task-from-agent

5. 動作確認
   └─ VSCodeでエージェント・プロンプトを使用
```

---

## 📚 関連ドキュメント

- [デプロイスクリプト](./deploy_scripts.md)
- [GitHub Copilot 活用方針](./README.md)
