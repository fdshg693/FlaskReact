# AI Code Review Scripts (Python版)

Python実装のAIコードレビュー自動化ツールです。GitのdiffをOpenAI APIに送信し、詳細なコードレビューを生成します。

## 特徴

- 🚀 **自動化**: Git diffの生成からAIレビューまで一括実行
- 🔒 **セキュア**: .envファイルで環境変数を管理
- 🌐 **クロスプラットフォーム**: Windows/Linux/macOS対応
- 🧪 **テスト済み**: 包括的な単体テスト・統合テスト
- ⚡ **高速**: 並行処理による効率的な実行

## 必要要件

- Python 3.8以上
- Git
- OpenAI APIキー

## インストール

### 1. リポジトリのクローン

```bash
cd scripts/ai_review
```

### 2. 仮想環境の作成（推奨）

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

プロジェクトルートまたはスクリプトディレクトリに `.env` ファイルを作成します：

```env
# 必須
OPENAI_API_KEY=your-openai-api-key-here

# オプション（デフォルト値）
AI_MODEL=gpt-4o
MAX_TOKENS=10000
TEMPERATURE=0.1
```

## 使い方

### 基本的な使用方法

```bash
# mainブランチとの差分をレビュー
python ai-review_orchestrator.py

# 特定のブランチとの差分をレビュー
python ai-review_orchestrator.py develop
# または
python ai-review_orchestrator.py -b develop
```

### 詳細オプション

```bash
# 詳細ログ表示
python ai-review_orchestrator.py -v

# 最小限の出力のみ
python ai-review_orchestrator.py -q

# カスタムモデル指定
python ai-review_orchestrator.py --model gpt-3.5-turbo

# カスタムプロンプトを使用
python ai-review_orchestrator.py --prompt-file custom_prompt.txt

# 大きなdiffを制限（最初の500行のみ）
python ai-review_orchestrator.py --max-lines 500

# 組み合わせ
python ai-review_orchestrator.py -b develop --model gpt-4o --max-lines 1000 -v
```

### 個別スクリプトの実行

#### Diff生成のみ

```bash
python generate_diff.py -b main -o output/diff.patch
```

#### AIレビューのみ

```bash
python generate_ai_review.py tmp/diff.patch -o output/review.md
```

## 出力ファイル

- `tmp/diff.patch` - 生成されたGit diff
- `tmp/ai_review_output.md` - AIによるコードレビュー

## コマンドラインオプション

### ai-review_orchestrator.py

| オプション          | 説明                           | デフォルト              |
| ------------------- | ------------------------------ | ----------------------- |
| `base_branch`       | 比較対象のベースブランチ       | main/master（自動検出） |
| `-b, --base-branch` | ベースブランチ（別の指定方法） | -                       |
| `-v, --verbose`     | 詳細ログを表示                 | False                   |
| `-q, --quiet`       | 最小限の出力                   | False                   |
| `--model`           | 使用するAIモデル               | gpt-4o                  |
| `--prompt-file`     | カスタムプロンプトファイル     | -                       |
| `--max-lines`       | 処理する最大行数               | 無制限                  |

### generate_diff.py

| オプション          | 説明                     | デフォルト     |
| ------------------- | ------------------------ | -------------- |
| `-b, --base-branch` | 比較対象のベースブランチ | main           |
| `-o, --output`      | 出力ファイルパス         | tmp/diff.patch |
| `-v, --verbose`     | 詳細ログを表示           | False          |

### generate_ai_review.py

| オプション          | 説明                       | デフォルト              |
| ------------------- | -------------------------- | ----------------------- |
| `diff_file`         | レビュー対象のdiffファイル | tmp/diff.patch          |
| `-o, --output`      | 出力ファイルパス           | tmp/ai_review_output.md |
| `-p, --prompt-file` | カスタムプロンプトファイル | -                       |
| `--model`           | 使用するAIモデル           | gpt-4o                  |
| `--max-retries`     | 最大リトライ回数           | 3                       |
| `--retry-delay`     | リトライ間隔（秒）         | 5                       |
| `--max-lines`       | 処理する最大行数           | 無制限                  |
| `-v, --verbose`     | 詳細ログを表示             | False                   |

## 環境変数

| 変数名           | 説明                                   | 必須 | デフォルト |
| ---------------- | -------------------------------------- | ---- | ---------- |
| `OPENAI_API_KEY` | OpenAI APIキー                         | ✅   | -          |
| `AI_MODEL`       | 使用するAIモデル                       | ❌   | gpt-4o     |
| `MAX_TOKENS`     | 最大トークン数                         | ❌   | 10000      |
| `TEMPERATURE`    | 生成の温度パラメータ                   | ❌   | 0.1        |
| `PROJECT_ROOT`   | プロジェクトルートパス                 | ❌   | 自動検出   |
| `PR_BASE_REF`    | PRのベースブランチ（GitHub Actions用） | ❌   | -          |
| `INPUT_TARGET`   | ターゲットブランチ（GitHub Actions用） | ❌   | -          |

## テスト

### テストの実行

```bash
# すべてのテストを実行
pytest

# 詳細な出力
pytest -v

# カバレッジ付き
pytest --cov=. --cov-report=html

# 特定のテストファイルのみ
pytest tests/test_config.py

# 特定のテストクラスのみ
pytest tests/test_config.py::TestReviewConfig

# 特定のテストメソッドのみ
pytest tests/test_config.py::TestReviewConfig::test_init_with_valid_env
```

### テストカバレッジの確認

```bash
pytest --cov=. --cov-report=term-missing
```

カバレッジレポートは `htmlcov/index.html` で確認できます。

## プロジェクト構造

```
ai_review/
├── config.py                    # 設定管理
├── generate_diff.py             # Diff生成
├── generate_ai_review.py        # AIレビュー生成
├── ai-review_orchestrator.py   # 統合スクリプト
├── requirements.txt             # 依存パッケージ
├── README.md                    # このファイル
├── tests/                       # テストディレクトリ
│   ├── __init__.py
│   ├── conftest.py             # 共通フィクスチャ
│   ├── test_config.py          # config.py のテスト
│   ├── test_generate_diff.py   # generate_diff.py のテスト
│   ├── test_generate_ai_review.py  # generate_ai_review.py のテスト
│   └── test_orchestrator.py    # orchestrator のテスト
├── tmp/                         # 一時ファイル
│   ├── diff.patch
│   └── ai_review_output.md
└── tmpdocs/
    └── plans.md                 # 実装計画
```

## トラブルシューティング

### エラー: `OPENAI_API_KEY が設定されていません`

`.env` ファイルにAPIキーが正しく設定されているか確認してください。

```env
OPENAI_API_KEY=sk-...
```

APIキーの取得方法：

1. [OpenAI Platform](https://platform.openai.com/api-keys)にアクセス
2. APIキーを生成
3. `.env`ファイルに設定

### エラー: `Gitリポジトリが見つかりません`

Gitリポジトリ内で実行しているか確認してください。

```bash
git status
```

初期化されていない場合：

```bash
git init
git remote add origin <repository-url>
```

### エラー: `ベースブランチのフェッチに失敗しました`

指定したブランチがリモートに存在するか確認してください。

```bash
# リモートブランチ一覧を確認
git branch -r

# 特定のブランチをフェッチ
git fetch origin main
```

### 大きなdiffでタイムアウト

`--max-lines` オプションで処理する行数を制限してください。

```bash
python ai-review_orchestrator.py --max-lines 500
```

### OpenAI APIレート制限エラー

自動的にリトライされますが、頻発する場合は以下を試してください：

```bash
# リトライ回数と間隔を調整
python generate_ai_review.py --max-retries 5 --retry-delay 10 tmp/diff.patch
```

または、APIプランをアップグレードしてください。

### Python環境のエラー

```bash
# Python バージョン確認
python --version  # 3.8以上が必要

# 依存パッケージの再インストール
pip install --upgrade -r requirements.txt

# 仮想環境の再作成
deactivate  # 仮想環境を無効化
rm -rf .venv  # 仮想環境を削除
python -m venv .venv  # 再作成
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Bashスクリプトが動作しない（Linux/macOS）

実行権限を付与してください：

```bash
chmod +x generate-diff.sh
chmod +x generate-ai-review.sh
chmod +x ai-review_orchestrator.sh
```

## よくある質問（FAQ）

### Q1: Bash版とPython版の違いは何ですか？

**A**: 機能的には同等ですが、Python版には以下の利点があります：

- ✅ Windows環境での実行容易性
- ✅ 包括的なエラーハンドリング
- ✅ 詳細なログ出力
- ✅ 豊富なコマンドラインオプション
- ✅ テストコードによる品質保証
- ✅ クラスベースの保守しやすい設計

詳細は[互換性ガイド](tmpdocs/COMPATIBILITY.md)を参照してください。

### Q2: 既存のBashスクリプトは使えますか？

**A**: はい、使えます。Bashスクリプトは後方互換性のためのラッパーとして機能し、内部でPython版を呼び出します。既存のワークフローはそのまま動作します。

```bash
# Bashラッパー経由（既存の方法）
bash ai-review_orchestrator.sh

# Python版を直接実行（推奨）
python ai_review_orchestrator.py
```

### Q3: どのAIモデルを使用していますか？

**A**: デフォルトは `gpt-4o` です。環境変数またはコマンドラインオプションで変更できます：

```bash
# 環境変数で指定
export AI_MODEL="gpt-4-turbo"
python ai_review_orchestrator.py

# コマンドラインオプションで指定
python ai_review_orchestrator.py --model gpt-3.5-turbo
```

利用可能なモデルは[OpenAI Models](https://platform.openai.com/docs/models)を参照してください。

### Q4: レビューの品質を向上させるには？

**A**: 以下の方法を試してください：

1. **カスタムプロンプトを使用**：

   ```bash
   python ai_review_orchestrator.py --prompt-file custom_prompt.txt
   ```

2. **より高性能なモデルを使用**：

   ```bash
   python ai_review_orchestrator.py --model gpt-4
   ```

3. **差分を適切なサイズに制限**：
   ```bash
   # 大きすぎる差分は品質が低下する可能性があります
   python ai_review_orchestrator.py --max-lines 1000
   ```

### Q5: コストはどのくらいかかりますか？

**A**: OpenAI APIの料金はモデルとトークン数によって異なります。

- **gpt-3.5-turbo**: 安価（$0.001/1K tokens）
- **gpt-4o**: 中程度（$0.005/1K tokens）
- **gpt-4**: 高価（$0.03/1K tokens）

コスト削減のヒント：

- `--max-lines`で処理行数を制限
- 頻繁に変更される部分のみレビュー
- プロンプトを簡潔に保つ

最新の料金は[OpenAI Pricing](https://openai.com/pricing)を確認してください。

### Q6: GitHub Actionsで使用できますか？

**A**: はい、簡単に統合できます。[GitHub Actionsとの統合](#github-actionsとの統合)セクションを参照してください。

```yaml
- name: Run AI Review
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    cd scripts/ai_review
    python ai_review_orchestrator.py -q
```

### Q7: プライベートリポジトリで使用しても安全ですか？

**A**: コードは OpenAI API に送信されます。以下の点を考慮してください：

- ✅ OpenAIは30日後にデータを削除（APIポリシー）
- ⚠️ 機密情報を含むコードには注意が必要
- 💡 `.gitignore`で機密ファイルを除外
- 💡 環境変数やシークレットは `.env` で管理

詳細は[OpenAI Data Usage Policy](https://openai.com/policies/usage-policies)を参照してください。

### Q8: 特定のファイルやディレクトリを除外できますか？

**A**: 現在、直接的な除外機能はありませんが、Gitの機能を利用できます：

```bash
# 特定のパスのみをコミット
git add specific/path/
git commit -m "Review target"

# または、.gitignoreで不要なファイルを除外
echo "vendor/" >> .gitignore
echo "node_modules/" >> .gitignore
```

### Q9: 複数のブランチを一度にレビューできますか？

**A**: 現在は1つのブランチずつですが、スクリプトを使って自動化できます：

```bash
# 複数ブランチをループ
for branch in feature-1 feature-2 feature-3; do
  python ai_review_orchestrator.py -b $branch
  mv tmp/ai_review_output.md reviews/review_${branch}.md
done
```

### Q10: レビュー結果をどこで確認できますか？

**A**: レビュー結果は `tmp/ai_review_output.md` に保存されます：

```bash
# ターミナルで表示
cat tmp/ai_review_output.md

# エディタで開く
code tmp/ai_review_output.md

# GitHub Actionsでアーティファクトとしてアップロード
# （ワークフローファイルの例を参照）
```

### Q11: Windowsで「bad interpreter」エラーが出る

**A**: Bashスクリプトの行末がCRLFになっている可能性があります：

```bash
# LFに変換（Git Bash）
dos2unix generate-diff.sh
dos2unix generate-ai-review.sh
dos2unix ai-review_orchestrator.sh

# または、Pythonを直接使用（推奨）
python ai_review_orchestrator.py
```

### Q12: テストはどこにありますか？

**A**: `tests/` ディレクトリに包括的なテストがあります：

```bash
# すべてのテストを実行
pytest

# カバレッジ付き
pytest --cov=. --cov-report=html

# 特定のテストのみ
pytest tests/test_orchestrator.py -v
```

テストカバレッジ: 主要機能100%達成

### Q13: カスタマイズしたいのですが、どこから始めればよいですか？

**A**: 以下のファイルを編集してください：

- **プロンプトのカスタマイズ**: `generate_ai_review.py` の `DEFAULT_PROMPT`
- **設定のカスタマイズ**: `config.py` の `ReviewConfig` クラス
- **ワークフローのカスタマイズ**: `ai_review_orchestrator.py` の `PRReviewOrchestrator` クラス

詳細は各ファイルのdocstringを参照してください。

### Q14: 移行ガイドはありますか？

**A**: はい、[互換性ガイド](tmpdocs/COMPATIBILITY.md)に詳細な移行手順があります：

- Bash版からPython版への移行手順
- 環境変数の互換性
- コマンドライン引数の違い
- トラブルシューティング

## 移行ガイド

Bash版からPython版への移行については、[互換性ガイド](tmpdocs/COMPATIBILITY.md)を参照してください。

主な移行ステップ：

1. **Python環境の準備**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Bashラッパー経由でテスト**

   ```bash
   bash ai-review_orchestrator.sh
   ```

3. **Python版を直接実行**

   ```bash
   python ai_review_orchestrator.py -v
   ```

4. **ワークフローを更新**（オプション）
   - GitHub ActionsなどでPython版を直接呼び出すように変更
   - より詳細なオプションを活用

## カスタムプロンプトの作成

カスタムプロンプトファイル（例: `custom_prompt.txt`）を作成：

```
あなたは経験豊富なソフトウェアエンジニアです。
以下のコードdiffを日本語で詳細にレビューしてください：

1. コード品質
2. セキュリティ
3. パフォーマンス
4. 改善提案

コードdiff:
```

実行：

```bash
python ai-review_orchestrator.py --prompt-file custom_prompt.txt
```

## GitHub Actionsとの統合

`.github/workflows/ai-review.yml` の例：

```yaml
name: AI Code Review

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
          python ai-review_orchestrator.py -q

      - name: Upload Review
        uses: actions/upload-artifact@v3
        with:
          name: ai-review
          path: scripts/ai_review/tmp/ai_review_output.md
```

## ライセンス

このプロジェクトは内部使用のためのものです。

## 開発

### 開発環境のセットアップ

```bash
# 開発用パッケージのインストール
pip install -r requirements.txt

# pre-commitのセットアップ（オプション）
pip install pre-commit
pre-commit install
```

### コードフォーマット

```bash
# blackでフォーマット
black *.py tests/*.py

# ruffでlint
ruff check *.py tests/*.py
```

### 型チェック

```bash
mypy *.py
```

## 変更履歴

### v1.0.0 (2026-01-17)

- ✨ Python版の初回リリース
- ✅ 包括的なテストスイート
- 📚 完全なドキュメント
- 🐛 エラーハンドリングの改善
- 🎨 ユーザーフレンドリーなCLI

## サポート

問題や質問がある場合は、プロジェクトの担当者に連絡してください。

## 参考資料

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [GitPython Documentation](https://gitpython.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
