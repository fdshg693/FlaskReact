# Bash版とPython版の互換性ガイド

## 環境変数の互換性

### 完全互換（同じ変数名・同じ意味）

| 変数名           | 説明                       | Bash版 | Python版 | 備考                          |
| ---------------- | -------------------------- | ------ | -------- | ----------------------------- |
| `OPENAI_API_KEY` | OpenAI APIキー             | ✅必須 | ✅必須   | -                             |
| `AI_MODEL`       | 使用するAIモデル           | ✅     | ✅       | デフォルト値が異なる場合あり  |
| `MAX_TOKENS`     | 最大トークン数             | ✅     | ✅       | -                             |
| `TEMPERATURE`    | 生成の温度パラメータ       | ✅     | ✅       | -                             |
| `REVIEW_PROMPT`  | カスタムレビュープロンプト | ✅     | ✅       | Python版は--prompt-fileも対応 |

### GitHub Actions固有（部分互換）

| 変数名          | 説明                       | Bash版 | Python版 | 備考                 |
| --------------- | -------------------------- | ------ | -------- | -------------------- |
| `GITHUB_OUTPUT` | GitHub Actions出力ファイル | ✅必須 | ❌不要   | Python版は使用しない |
| `PR_BASE_REF`   | PRのベースブランチ         | ✅     | ✅       | 両方とも対応         |
| `INPUT_TARGET`  | ターゲットブランチ         | ✅     | ✅       | 両方とも対応         |

### Python版のみ

| 変数名         | 説明                   | デフォルト | 備考         |
| -------------- | ---------------------- | ---------- | ------------ |
| `PROJECT_ROOT` | プロジェクトルートパス | 自動検出   | 通常設定不要 |

## デフォルト値の違い

### Bash版のデフォルト値

```bash
AI_MODEL="${AI_MODEL:-gpt-4.1}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
TEMPERATURE="${TEMPERATURE:-0.1}"
BASE_BRANCH="${1:-main}"
```

### Python版のデフォルト値

```python
AI_MODEL = "gpt-4o"          # Bash版: gpt-4.1
MAX_TOKENS = 10000
TEMPERATURE = 0.1
BASE_BRANCH = "main" または "master"（自動検出）
```

**注意**: Python版では最新のモデル名 `gpt-4o` がデフォルトです。Bash版の `gpt-4.1` を使用したい場合は環境変数で明示的に指定してください。

## コマンドライン引数の互換性

### generate-diff.sh / generate_diff.py

**Bash版:**

```bash
# 環境変数のみで制御
export INPUT_TARGET="develop"
./generate-diff.sh
```

**Python版:**

```bash
# コマンドライン引数で制御（推奨）
python generate_diff.py -b develop
```

**互換性**: Bashラッパーが環境変数を引数に変換するため、既存のワークフローはそのまま動作します。

### generate-ai-review.sh / generate_ai_review.py

**Bash版:**

```bash
# 位置引数
./generate-ai-review.sh tmp/diff.patch
```

**Python版:**

```bash
# 位置引数（互換）
python generate_ai_review.py tmp/diff.patch

# 名前付き引数（推奨）
python generate_ai_review.py tmp/diff.patch --model gpt-4o --max-lines 1000
```

**互換性**: Bashラッパーが環境変数を引数に変換するため、既存のワークフローはそのまま動作します。

### ai-review_orchestrator.sh / ai_review_orchestrator.py

**Bash版:**

```bash
# ベースブランチのみ指定
./ai-review_orchestrator.sh develop
```

**Python版:**

```bash
# ベースブランチ指定（互換）
python ai_review_orchestrator.py develop

# 名前付き引数（推奨）
python ai_review_orchestrator.py -b develop -v --model gpt-4o
```

**互換性**: Bashラッパーが引数を適切に渡すため、既存のワークフローはそのまま動作します。

## ファイル出力の互換性

### 出力ファイルパス

| ファイル                  | Bash版 | Python版 | 互換性 |
| ------------------------- | ------ | -------- | ------ |
| `tmp/diff.patch`          | ✅     | ✅       | ✅     |
| `tmp/ai_review_output.md` | ✅     | ✅       | ✅     |

### 出力フォーマット

- **diff.patch**: 完全互換。同じGitコマンドを使用。
- **ai_review_output.md**: 完全互換。同じプロンプトで同じレビューを生成。

## GitHub Actions統合の互換性

### Bash版の例

```yaml
- name: Run AI Review
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    PR_BASE_REF: ${{ github.base_ref }}
  run: |
    cd scripts/ai_review
    bash generate-diff.sh
    bash generate-ai-review.sh tmp/diff.patch
```

### Python版の例（直接実行）

```yaml
- name: Run AI Review
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    PR_BASE_REF: ${{ github.base_ref }}
  run: |
    cd scripts/ai_review
    python ai_review_orchestrator.py -q
```

### Python版の例（Bashラッパー経由）

```yaml
- name: Run AI Review
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    PR_BASE_REF: ${{ github.base_ref }}
  run: |
    cd scripts/ai_review
    bash ai-review_orchestrator.sh  # 既存のワークフローと互換
```

**互換性**: 既存のGitHub Actionsワークフローは変更不要で動作します。

## 移行時の注意点

### 1. Python環境の準備

```bash
# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化（Windows）
.venv\Scripts\activate

# 仮想環境の有効化（Linux/macOS）
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 環境変数の確認

既存の `.env` ファイルはそのまま使用できます。AIモデルを更新する場合は以下を追加：

```env
# 最新モデルを使用（Python版のデフォルト）
AI_MODEL=gpt-4o

# または従来のモデルを維持
AI_MODEL=gpt-4.1
```

### 3. 段階的な移行戦略

#### ステップ1: Bashラッパー経由でテスト

```bash
# 既存のBashスクリプトを実行（内部でPython版が動作）
bash ai-review_orchestrator.sh develop
```

#### ステップ2: Python版を直接実行してテスト

```bash
# Python版を直接実行
python ai_review_orchestrator.py develop -v
```

#### ステップ3: ワークフローを更新（オプション）

```yaml
# より詳細な制御が必要な場合は直接実行に変更
- name: Run AI Review
  run: |
    cd scripts/ai_review
    python ai_review_orchestrator.py -b ${{ github.base_ref }} --max-lines 1000 -v
```

## トラブルシューティング

### エラー: `python: command not found`

**原因**: Pythonがインストールされていない、またはPATHに含まれていない。

**解決方法**:

```bash
# Pythonのインストール確認
python --version
python3 --version

# Windowsの場合
winget install Python.Python.3.13

# Linuxの場合
sudo apt-get install python3 python3-pip

# macOSの場合
brew install python3
```

### エラー: `ModuleNotFoundError: No module named 'openai'`

**原因**: 依存パッケージがインストールされていない。

**解決方法**:

```bash
cd scripts/ai_review
pip install -r requirements.txt
```

### Bashラッパーが動作しない

**原因**: スクリプトに実行権限がない。

**解決方法**:

```bash
# Linux/macOS
chmod +x generate-diff.sh
chmod +x generate-ai-review.sh
chmod +x ai-review_orchestrator.sh

# Windows（Git Bashの場合）
# 通常は不要ですが、必要に応じて
bash ai-review_orchestrator.sh
```

### Python版とBash版で結果が異なる

**原因**: AIモデルのデフォルト値の違い。

**解決方法**:

```bash
# 環境変数で明示的に指定
export AI_MODEL="gpt-4.1"
python ai_review_orchestrator.py
```

## まとめ

- ✅ **環境変数**: 完全互換（`GITHUB_OUTPUT`を除く）
- ✅ **ファイル出力**: 完全互換
- ✅ **既存ワークフロー**: Bashラッパー経由で無変更で動作
- ⚠️ **AIモデル**: デフォルトが異なる（Python版: `gpt-4o`, Bash版: `gpt-4.1`）
- 💡 **推奨**: 段階的に移行し、Python版の直接実行に切り替える
