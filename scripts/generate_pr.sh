#!/usr/bin/env bash
# AI コードレビュー生成スクリプト
# このスクリプトは差分を生成し、OpenAI API を使用して AI コードレビューを作成します
#
# 使用方法:
#   ./scripts/generate_pr.sh [base_branch]
#
# 引数:
#   base_branch: オプション。比較対象のベースブランチ（デフォルト: main）
#
# 必要なファイル:
#   - .env: 環境変数ファイル（OPENAI_API_KEY が必要）
#
# 出力:
#   - tmp/diff.patch: 生成された差分ファイル
#   - tmp/ai_review_output.md: AI が生成したレビュー

#-e (errexit): コマンドがエラー(終了コード 0以外)を返したら即座にスクリプトを停止
#-u (nounset): 未定義の変数を参照した場合にエラーを出してスクリプトを停止
#-o pipefail: パイプライン内のいずれかのコマンドが失敗した場合にパイプライン全体を失敗とみなす
set -euo pipefail

# スクリプトのディレクトリを取得（絶対パス）
# pwd: 現在のディレクトリを表示(print working directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

echo "🚀 AI コードレビュープロセスを開始します"
echo "================================================"

# .env ファイルから環境変数を読み込み
# -f: ファイルが存在しないかどうかをチェックする条件式（存在しない場合はtrueを返す）
if [ ! -f .env ]; then
  echo "❌ エラー: プロジェクトルートに .env ファイルが見つかりません"
  echo "sample.env を参考に .env ファイルを作成してください"
  exit 1
fi

echo "📁 .env から環境変数を読み込み中..."
# .env ファイルから変数をエクスポート
# set -a を使用して全ての変数を自動的にエクスポート
set -a
source .env
set +a

# 必須の環境変数を検証
# -z: 文字列が空かどうかをチェックする条件式（空ならtrueを返す）
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ エラー: OPENAI_API_KEY が .env ファイルに設定されていません"
  exit 1
fi

# AI モデル設定のデフォルト値を設定
export AI_MODEL="${AI_MODEL:-gpt-4.1}"
export MAX_TOKENS="${MAX_TOKENS:-10000}"
export TEMPERATURE="${TEMPERATURE:-0.1}"

echo "✅ 環境変数の読み込み完了"
echo "   - モデル: $AI_MODEL"
echo "   - 最大トークン数: $MAX_TOKENS"
echo "   - Temperature: $TEMPERATURE"
echo ""

# ベースブランチを決定
BASE_BRANCH="${1:-main}"
echo "📊 比較対象のベースブランチ: $BASE_BRANCH"

# 既存のスクリプトとの互換性のために一時的な GITHUB_OUTPUT ファイルを作成
TEMP_OUTPUT=$(mktemp)
trap 'rm -f "$TEMP_OUTPUT"' EXIT
export GITHUB_OUTPUT="$TEMP_OUTPUT"

# ステップ 1: generate-diff.sh を使用して差分を生成
echo ""
echo "================================================"
echo "📝 ステップ 1: 差分を生成中..."
echo "================================================"

# generate-diff.sh に必要な変数を設定
export INPUT_TARGET="$BASE_BRANCH"

# generate-diff スクリプトを実行
if [ -f ".github/scripts/generate-diff.sh" ]; then
  bash .github/scripts/generate-diff.sh
else
  echo "❌ エラー: .github/scripts/generate-diff.sh が見つかりません"
  exit 1
fi

# 差分が生成されたか確認
if [ ! -f tmp/diff.patch ]; then
  echo "❌ エラー: tmp/diff.patch が作成されませんでした"
  exit 1
fi

# 変更があるか確認
# grep: テキストファイルやストリーム内でパターン（文字列や正規表現）に一致する行を検索。
# -d='=': 区切り文字を '=' に設定
# cut -f2: 区切り文字で分割した2番目のフィールドを抽出
HAS_CHANGES=$(grep "has_changes=" "$GITHUB_OUTPUT" | cut -d'=' -f2 || echo "false")

if [ "$HAS_CHANGES" = "false" ]; then
  echo "ℹ️ 変更が検出されませんでした。AI レビューをスキップします。"
  exit 0
fi

echo "✅ 差分の生成が完了しました"
echo ""

# ステップ 2: generate-ai-review.sh を使用して AI レビューを生成
echo "================================================"
echo "🤖 ステップ 2: AI レビューを生成中..."
echo "================================================"

# AI レビュースクリプトを実行
if [ -f ".github/scripts/generate-ai-review.sh" ]; then
  bash .github/scripts/generate-ai-review.sh tmp/diff.patch
else
  echo "❌ エラー: .github/scripts/generate-ai-review.sh が見つかりません"
  exit 1
fi

# レビューが生成されたか確認
if [ ! -f tmp/ai_review_output.md ]; then
  echo "❌ エラー: tmp/ai_review_output.md が作成されませんでした"
  exit 1
fi

echo ""
echo "================================================"
echo "✅ AI コードレビューが完了しました！"
echo "================================================"
echo ""
echo "📄 生成されたファイル:"
echo "   - tmp/diff.patch: $BASE_BRANCH と現在のブランチ間の Git 差分"
echo "   - tmp/ai_review_output.md: AI が生成したコードレビュー"
echo ""
echo "📖 レビューを確認するには:"
echo "   cat tmp/ai_review_output.md"
echo ""
