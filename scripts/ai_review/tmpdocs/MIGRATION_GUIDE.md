# Bash版からPython版への移行ガイド

このドキュメントは、既存のBash実装からPython実装へスムーズに移行するための詳細なガイドです。

## 📋 目次

1. [移行の理由](#移行の理由)
2. [互換性の概要](#互換性の概要)
3. [移行前の準備](#移行前の準備)
4. [段階的な移行手順](#段階的な移行手順)
5. [機能比較](#機能比較)
6. [トラブルシューティング](#トラブルシューティング)
7. [よくある質問](#よくある質問)

## 移行の理由

Python版への移行による主なメリット：

### ✅ クロスプラットフォーム対応

- Windows環境での実行が容易
- シェル環境に依存しない
- パスの扱いが統一的（`pathlib`使用）

### ✅ 保守性の向上

- クラスベースの設計
- 型ヒントによる安全性
- モジュール化された構造
- 包括的なテストスイート（61テストケース）

### ✅ エラーハンドリングの強化

- 詳細なエラーメッセージ
- 具体的な解決方法の提示
- リトライロジックの実装

### ✅ 機能の拡張

- 豊富なコマンドラインオプション
- カスタマイズ可能な設定
- 詳細なログ出力

## 互換性の概要

### 🟢 完全互換

以下は変更なしで動作します：

- ✅ 環境変数（`OPENAI_API_KEY`, `AI_MODEL`, `MAX_TOKENS`, `TEMPERATURE`）
- ✅ 入力ファイル（`tmp/diff.patch`）
- ✅ 出力ファイル（`tmp/ai_review_output.md`）
- ✅ 基本的なコマンドライン引数
- ✅ GitHub Actions統合

### 🟡 一部互換

以下は動作しますが、違いがあります：

- ⚠️ デフォルトAIモデル：`gpt-4.1` → `gpt-4o`
- ⚠️ `GITHUB_OUTPUT`：Bash版のみ必須、Python版は不要
- ⚠️ コマンドラインオプション：Python版の方が豊富

### 🔴 非互換

以下は使用できません：

- ❌ Bash固有の機能（`trap`, `mktemp`等）
- ❌ jq依存のJSON処理
- ❌ curl直接呼び出し

詳細は[COMPATIBILITY.md](COMPATIBILITY.md)を参照してください。

## 移行前の準備

### 1. システム要件の確認

```bash
# Python バージョン確認（3.8以上が必要）
python --version
# または
python3 --version

# Git がインストールされているか確認
git --version
```

必要に応じてインストール：

**Windows:**

```powershell
# Python のインストール
winget install Python.Python.3.13

# Gitのインストール
winget install Git.Git
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv git
```

**macOS:**

```bash
# Homebrew を使用
brew install python3 git
```

### 2. プロジェクトディレクトリへの移動

```bash
cd scripts/ai_review
```

### 3. Python仮想環境の作成

```bash
# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate
```

仮想環境が有効化されると、プロンプトに `(.venv)` が表示されます。

### 4. 依存パッケージのインストール

```bash
# 本番用パッケージのインストール
pip install -r requirements.txt

# インストールの確認
pip list
```

期待される出力：

```
openai       1.x.x
GitPython    3.x.x
python-dotenv 1.x.x
...
```

### 5. 環境変数の確認

既存の `.env` ファイルを確認：

```bash
# .envファイルの存在確認
ls .env
# または
cat .env
```

必要な内容：

```env
# 必須
OPENAI_API_KEY=sk-...

# オプション（デフォルトは以下の通り）
AI_MODEL=gpt-4o
MAX_TOKENS=10000
TEMPERATURE=0.1
```

`.env` ファイルがない場合は作成：

```bash
# Windowsの場合
echo OPENAI_API_KEY=sk-your-key-here > .env

# Linux/macOSの場合
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## 段階的な移行手順

### ステージ1: 検証フェーズ（Bashラッパー経由）

まず、Bashラッパー経由でPython実装をテストします。

```bash
# 既存のBashスクリプトを実行（内部でPython版が呼ばれる）
bash ai-review_orchestrator.sh

# または、特定のブランチを指定
bash ai-review_orchestrator.sh develop
```

**期待される動作:**

- ⚠️ ラッパーの警告メッセージが表示される
- ✅ Python版が内部で実行される
- ✅ 既存と同じ出力ファイルが生成される

**確認項目:**

- [ ] エラーなく実行完了
- [ ] `tmp/diff.patch` が生成されている
- [ ] `tmp/ai_review_output.md` が生成されている
- [ ] レビュー内容が適切

### ステージ2: Python直接実行フェーズ

Python版を直接実行してテストします。

```bash
# Python版を直接実行
python ai_review_orchestrator.py

# 詳細ログ付き
python ai_review_orchestrator.py -v

# 特定のブランチを指定
python ai_review_orchestrator.py -b develop

# 複数のオプションを組み合わせ
python ai_review_orchestrator.py -b develop --model gpt-4o --max-lines 1000 -v
```

**新機能の活用:**

```bash
# 処理する行数を制限
python ai_review_orchestrator.py --max-lines 500

# カスタムモデルを使用
python ai_review_orchestrator.py --model gpt-3.5-turbo

# カスタムプロンプトを使用
python ai_review_orchestrator.py --prompt-file custom_prompt.txt

# 最小限の出力
python ai_review_orchestrator.py -q
```

**確認項目:**

- [ ] コマンドが正常に実行される
- [ ] ログが適切に出力される
- [ ] エラーメッセージが理解しやすい
- [ ] レビュー品質が維持されている

### ステージ3: CI/CD統合フェーズ

GitHub Actionsやその他のCI/CDパイプラインを更新します。

#### 既存のワークフロー（Bash版）

```yaml
name: AI Code Review (Old)

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Run AI Review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          PR_BASE_REF: ${{ github.base_ref }}
        run: |
          cd scripts/ai_review
          bash ai-review_orchestrator.sh
```

#### 移行後のワークフロー（Python版）

```yaml
name: AI Code Review (New)

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          cd scripts/ai_review
          pip install -r requirements.txt

      - name: Run AI Review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          PR_BASE_REF: ${{ github.base_ref }}
        run: |
          cd scripts/ai_review
          python ai_review_orchestrator.py -q

      - name: Upload Review
        uses: actions/upload-artifact@v3
        with:
          name: ai-review
          path: scripts/ai_review/tmp/ai_review_output.md

      - name: Comment PR (optional)
        if: success()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('scripts/ai_review/tmp/ai_review_output.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '## 🤖 AI Code Review\n\n' + review
            });
```

**移行オプション:**

1. **段階的移行（推奨）:**
   - 新しいワークフローを追加
   - 古いワークフローと並行実行
   - 動作確認後に古いワークフローを削除

2. **即時移行:**
   - 既存のワークフローを直接更新
   - テストブランチで事前検証

### ステージ4: 完全移行フェーズ

すべての環境でPython版に切り替えます。

**チェックリスト:**

- [ ] すべての開発者がPython環境をセットアップ完了
- [ ] CI/CDパイプラインが正常動作
- [ ] ドキュメントが更新済み
- [ ] チーム内で周知完了
- [ ] 旧Bashスクリプトに非推奨マークを追加

**非推奨マークの追加:**

各Bashスクリプトの先頭に追加：

```bash
#!/usr/bin/env bash
# ⚠️  DEPRECATED: This script is deprecated. Please use the Python version instead.
# For migration guide, see: tmpdocs/MIGRATION_GUIDE.md
#
# Bash版からPython版へ移行してください：
#   bash ai-review_orchestrator.sh  → python ai_review_orchestrator.py
#
# This is a backward compatibility wrapper that calls the Python version.
```

## 機能比較

### コマンド対応表

| 機能               | Bash版                           | Python版                                       |
| ------------------ | -------------------------------- | ---------------------------------------------- |
| 基本実行           | `bash ai-review_orchestrator.sh` | `python ai_review_orchestrator.py`             |
| ブランチ指定       | `bash script.sh develop`         | `python script.py develop` または `-b develop` |
| 詳細ログ           | -（環境変数のみ）                | `-v` または `--verbose`                        |
| 最小出力           | -                                | `-q` または `--quiet`                          |
| モデル指定         | `export AI_MODEL=gpt-4`          | `--model gpt-4` または環境変数                 |
| 行数制限           | -                                | `--max-lines 500`                              |
| カスタムプロンプト | `export REVIEW_PROMPT="..."`     | `--prompt-file custom.txt` または環境変数      |
| リトライ設定       | -（固定）                        | `--max-retries 5 --retry-delay 10`             |

### 機能の詳細比較

#### 1. Diff生成

**Bash版:**

```bash
export INPUT_TARGET="develop"
bash generate-diff.sh
```

**Python版:**

```bash
# 基本
python generate_diff.py -b develop

# オプション付き
python generate_diff.py -b develop -o custom/path.patch -v
```

#### 2. AIレビュー生成

**Bash版:**

```bash
export AI_MODEL="gpt-4"
export MAX_TOKENS=10000
bash generate-ai-review.sh tmp/diff.patch
```

**Python版:**

```bash
# 基本
python generate_ai_review.py tmp/diff.patch

# オプション付き
python generate_ai_review.py tmp/diff.patch \
  --model gpt-4 \
  --max-lines 1000 \
  --max-retries 5 \
  --retry-delay 10 \
  -v
```

#### 3. 統合実行

**Bash版:**

```bash
bash ai-review_orchestrator.sh develop
```

**Python版:**

```bash
# シンプル
python ai_review_orchestrator.py develop

# フル機能
python ai_review_orchestrator.py \
  -b develop \
  --model gpt-4o \
  --max-lines 1000 \
  --prompt-file custom_prompt.txt \
  -v
```

## トラブルシューティング

### 問題1: `ModuleNotFoundError: No module named 'openai'`

**原因:** 依存パッケージがインストールされていない

**解決方法:**

```bash
# 仮想環境が有効化されているか確認
which python  # Linux/macOS
where python  # Windows

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 問題2: `python: command not found`

**原因:** Pythonがインストールされていない、またはPATHに含まれていない

**解決方法:**

```bash
# Python3 を試す
python3 --version

# インストールが必要な場合
# Windows
winget install Python.Python.3.13

# Linux
sudo apt-get install python3 python3-pip

# macOS
brew install python3
```

### 問題3: Bashラッパーが「bad interpreter」エラー

**原因:** 行末がCRLFになっている（Windowsの問題）

**解決方法:**

```bash
# Git Bashで行末を修正
dos2unix *.sh

# または、Git設定を変更
git config core.autocrlf input

# または、Pythonを直接使用（推奨）
python ai_review_orchestrator.py
```

### 問題4: レビュー結果がBash版と異なる

**原因:** AIモデルのデフォルト値が異なる

**解決方法:**

```bash
# 環境変数で明示的に指定
export AI_MODEL="gpt-4.1"
python ai_review_orchestrator.py

# または.envファイルに追加
echo "AI_MODEL=gpt-4.1" >> .env
```

### 問題5: 仮想環境が有効化できない（Windows PowerShell）

**原因:** 実行ポリシーの制限

**解決方法:**

```powershell
# 現在のセッションのみ実行ポリシーを変更
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 仮想環境を有効化
.venv\Scripts\Activate.ps1
```

## よくある質問

### Q1: 移行にはどれくらいの時間がかかりますか？

**A:** 環境によりますが、通常：

- 準備（Python環境構築）: 10-30分
- 検証（Bashラッパー経由）: 5-10分
- 直接実行テスト: 5-10分
- CI/CD更新: 15-30分

**合計: 約1時間**

### Q2: チーム全体で移行する必要がありますか？

**A:** いいえ。Bashラッパーがあるため、段階的な移行が可能です：

- ✅ 一部の開発者はPython版を直接使用
- ✅ 他の開発者はBashラッパー経由で使用
- ✅ CI/CDは先にPython版に移行可能

### Q3: Bash版を削除してもいいですか？

**A:** チーム全員がPython版に移行し、以下を確認してから：

- [ ] すべての開発環境でPython版が動作
- [ ] CI/CDがPython版を使用
- [ ] ドキュメントが更新済み
- [ ] チーム内で合意形成

削除前に非推奨マークを付けて数週間様子を見ることを推奨します。

### Q4: Python版は本当にBash版と同じ機能ですか？

**A:** はい、コア機能は同等です。さらに以下が追加されています：

- ✅ より豊富なコマンドラインオプション
- ✅ 詳細なエラーメッセージ
- ✅ リトライロジック
- ✅ 行数制限機能
- ✅ 包括的なテスト

### Q5: パフォーマンスに違いはありますか？

**A:** 実用上の違いはありません：

- ⚡ 起動時間: Python版がわずかに遅い（0.1-0.2秒）
- ⚡ 実行時間: ほぼ同等（API呼び出しが支配的）
- ⚡ メモリ使用: 同等

### Q6: 古いPythonバージョン（3.7以前）でも動きますか？

**A:** いいえ、Python 3.8以上が必要です。理由：

- 型ヒント機能（`from __future__ import annotations`）
- 一部のライブラリの最小要件
- pathlib の完全サポート

アップグレードを推奨します：

```bash
# 現在のバージョン確認
python --version

# アップグレード方法はOSごとに異なります
```

## まとめ

### 移行のメリット

✅ **即時の利点:**

- Windows環境での実行容易性
- 詳細なエラーメッセージ
- 豊富なコマンドラインオプション

✅ **中長期的な利点:**

- 保守性の向上
- テストによる品質保証
- 機能拡張の容易性

### 推奨される移行パス

1. **Week 1**: 環境構築とBashラッパー経由での検証
2. **Week 2**: Python版の直接実行に慣れる
3. **Week 3**: CI/CDの更新
4. **Week 4**: チーム全体での採用とBash版の非推奨化

### サポート

移行に関する質問や問題がある場合：

1. このガイドの[トラブルシューティング](#トラブルシューティング)を確認
2. [README.md](../README.md)のFAQを確認
3. [互換性ガイド](COMPATIBILITY.md)を参照
4. プロジェクト担当者に連絡

**Happy Migrating! 🚀**
