#!/usr/bin/env bash
# Git差分生成スクリプト
# このスクリプトは現在のブランチとベースブランチの間の差分を生成します
#
# 必須の環境変数:
#   - GITHUB_OUTPUT: GitHub Actionsの出力ファイルパス
#
# オプションの環境変数:
#   - PR_BASE_REF: プルリクエストのベースブランチ参照
#   - INPUT_TARGET: ワークフロー入力からのターゲットブランチ
#
# 引数:
#   なし (環境変数を使用)
#
# 出力:
#   - tmp/diff.patch: 生成された差分ファイル
#   - GITHUB_OUTPUT設定: has_changes=true/false

set -euo pipefail
# 注記: Bash/シェルスクリプトの用語:
# - mktemp: ユニークな名前の一時ファイルを作成し、そのファイルパスを返す。
#           ここではプロンプトとAPIペイロード用の一時ファイルを作成するために使用。
#           例: TEMP_PROMPT=$(mktemp) は /tmp/tmp.XXXXXXXXXX を作成
# 
# - trap: スクリプト終了時（正常終了またはエラー時）に実行するコマンドを登録。
#         構文: trap 'command' EXIT
#         ここでは使用後の一時ファイル削除を確実にするために使用。
#         例: trap 'rm -f "$TEMP_PROMPT"' EXIT は終了時に $TEMP_PROMPT を削除
#
# - ${VARIABLE:-default}: デフォルト値を持つパラメータ展開。
#         VARIABLEが設定されていて空でない場合はそれを返し、そうでなければ'default'を返す。
#         構文: ${VARIABLE:-default_value}
#         例: ${OPENAI_API_KEY:-} はOPENAI_API_KEYが未設定なら""を返す。
#         ここでは環境変数が設定されているか安全にチェックするために使用。

# 必須環境変数の検証
if [ -z "${GITHUB_OUTPUT:-}" ]; then
  echo "❌ Error: GITHUB_OUTPUT environment variable is not set"
  exit 1
fi

# 比較用のベースブランチを決定
PR_BASE_REF="${PR_BASE_REF:-}"
INPUT_TARGET="${INPUT_TARGET:-}"

if [ -n "$PR_BASE_REF" ] && [ "$PR_BASE_REF" != 'null' ]; then
  BASE_BRANCH="$PR_BASE_REF"
elif [ -n "$INPUT_TARGET" ] && [ "$INPUT_TARGET" != 'null' ]; then
  BASE_BRANCH="$INPUT_TARGET"
else
  BASE_BRANCH='main'
fi

echo "Comparing against base branch: $BASE_BRANCH"

# ベースブランチをフェッチ
# git branch -r: リモート追跡ブランチの一覧を表示
git fetch origin "$BASE_BRANCH" || {
  echo "❌ エラー: ベースブランチ'$BASE_BRANCH'のフェッチに失敗しました"
  echo "利用可能なブランチ:"
  git branch -r
  exit 1
}

# 差分を生成
# git diff "A...B": 3つのドット構文 - AとBの共通祖先(マージベース)からBまでの変更を表示
#                   プルリクエストで実際に追加された変更のみを抽出するのに最適
#                   (2つのドット"A..B"は単純な差分比較で、ベースブランチの変更も含まれる)
# --unified=3: 差分の前後3行のコンテキストを含める(デフォルト値、明示的に指定)
# --no-color: ANSIカラーコードを出力しない(ファイル保存やAPI送信用にプレーンテキスト化)
# --ignore-space-change: 空白のみの変更を無視(インデント調整などのノイズを除外)
# > tmp/diff.patch: 標準出力をファイルにリダイレクト
git diff "origin/$BASE_BRANCH...HEAD" \
  --unified=3 \
  --no-color \
  --ignore-space-change \
  > tmp/diff.patch

# 差分に内容があるかチェック
if [ ! -s tmp/diff.patch ]; then
  echo "ℹ️ 現在のブランチと$BASE_BRANCHの間に変更が検出されませんでした"
  echo "has_changes=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "has_changes=true" >> "$GITHUB_OUTPUT"
echo "✅ Generated diff file ($(wc -l < tmp/diff.patch) lines)"
