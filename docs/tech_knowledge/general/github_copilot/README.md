# GitHub Copilot 概要

GitHub Copilotは、AIを活用したコーディングアシスタントです。コード補完、チャット、カスタマイズ可能なエージェント機能を提供します。

---

## 📖 主な機能

### 1. コード補完

エディタ上でリアルタイムにコード候補を提案します。

- コンテキストに応じたコード生成
- 関数・クラスの自動補完
- コメントからのコード生成

### 2. Copilot Chat

対話形式でコードの説明、生成、リファクタリングを依頼できます。

- コードの解説
- エラーの診断・修正提案
- テストコードの生成
- ドキュメント生成

### 3. カスタマイズ機能

プロジェクトに応じてCopilotの動作をカスタマイズできます。

| 機能 | 説明 | 詳細 |
|------|------|------|
| **エージェント** | 特定の役割を持つAIアシスタント | [agents.md](./agents.md) |
| **プロンプト** | 特定タスク用のテンプレート | [prompts.md](./prompts.md) |
| **Instructions** | 常時適用される指示 | [instructions.md](./instructions.md) |

---

## 🔧 設定ファイルの配置

GitHub Copilotのカスタマイズファイルは `.github/` ディレクトリに配置します。

```
.github/
├── agents/                    # エージェント定義
│   └── *.agent.md
├── prompts/                   # プロンプトテンプレート
│   └── *.prompt.md
├── copilot-instructions.md    # グローバル指示（オプション）
└── instructions/              # 個別指示（オプション）
    └── *.instruction.md
```

---

## 📚 関連ドキュメント

- [エージェント機能](./agents.md)
- [プロンプト機能](./prompts.md)
- [Instructions機能](./instructions.md)
