# GitHub Copilot / AI Prompts 説明

本プロジェクトで AI (Copilot Chat 等) を活用する際の "instructions" と "prompts" の役割 / 使い分け / 推奨フローをまとめます。

## 用語
- Instructions: モデルに「現在の開発コンテキスト(方針/スタイル/制約)」を伝えるテンプレ。コードと一緒に読み込ませる。
- Prompts: 具体的タスクを実行させるための指示テンプレ（レビュー/修正/近代化など）。

## ディレクトリ構成
```
.github/
  instructions/   # 環境やスタイルを教える
  prompts/        # レビュー/修正等の具体アクション
  copilot-instructions.md  # 上位レベルの設計 & 開発パターン
  explanation.md  # ← 本ファイル
```

## Instructions 一覧
| ファイル | 目的 | 主な内容 | 使い方 |
|----------|------|----------|--------|
| modern.instructions.md | Pythonモダンスタイル共有 | pathlib, 型, loguru, pydantic 等 | 対象Pythonファイル + このファイルをコンテキストへ |
| react.instructions.md | React(CDN)前提の記法共有 | useState/useEffectのインポート方法等 | 対象JS/JSXファイルと一緒に読み込ませる |

### 推奨: 最初に instructions を読み込ませてから個別 Prompts を投げる

## Prompts 一覧（Python）
| ファイル | 目的 | 事前条件 | 出力例 |
|----------|------|----------|--------|
| python_review.prompt.md | コードレビュー生成 | 対象 .py ファイル | review/ 配下にレビュー Markdown |
| python_fix.prompt.md | レビュー指摘の優先修正 | reviewファイル + 対象コード | 修正差分提案 |
| python_modernize.prompt.md | 最新構文/ライブラリ導入 | 対象コード | 改良提案/置換コード |
| python_rename.prompt.md | 名前改善リファクタ | 対象コード | リネーム案/理由 |
| python_complex.prompt.md | 複雑コードの簡素化 | 対象コード | リファクタ候補 |
| prompt_refine.prompt.md | 既存プロンプト自体の改善 | 改善したいプロンプト | 改訂版 |
| summary.prompt.md | コード/差分要約 | 対象コード/PR差分 | 要約テキスト |
| _japanese.prompt.md | 日本語回答指示 | 他プロンプトと組合せ可 | 日本語出力 |

## Prompts 一覧（React）
| ファイル | 目的 | 事前条件 |
|----------|------|----------|
| react_review.prompt.md | Reactコードレビュー | 対象JSX/JS + instructions/react |
| react_fix.prompt.md | レビュー内容の修正適用 | reviewファイル + 対象コード |

## 実行フロー例
### 1. Pythonコード改善
1. VSCode Chat で対象 `.py` をエディタで開きつつ `modern.instructions.md` を追加
2. `/python_review` を実行 → `src/.../review/xxx_review.md` 生成
3. `/python_fix` で最重要指摘を適用
4. 必要なら `/python_modernize` や `/python_rename`

### 2. Reactコード改善
1. 対象 JSX + `react.instructions.md` をコンテキスト投入
2. `/react_review` → reviewファイル作成
3. `/react_fix` で修正案

## review ファイル配置ルール
- 原則: 対象コードと同階層に `review/` ディレクトリ（なければ新規作成）
- 命名: `<元ファイル名>_review.md`
- 複数回レビュー: 既存レビューを追記 or バージョン番号付与
