#!/usr/bin/env bash
# AIコードレビュー生成スクリプト
# このスクリプトは差分をOpenAI APIに送信し、コードレビューを生成します
# 
# 必須の環境変数:
#   - OPENAI_API_KEY: OpenAI APIキー
#   - AI_MODEL: 使用するモデル (例: gpt-4)
#   - MAX_TOKENS: レスポンスの最大トークン数
#   - TEMPERATURE: モデルの温度パラメータ
#   - GITHUB_OUTPUT: GitHub Actionsの出力ファイルパス
#
# オプションの環境変数:
#   - REVIEW_PROMPT: カスタムレビュープロンプト (未設定の場合はデフォルトを使用)
#
# 引数:
#   $1: 差分ファイルのパス

#-e (errexit): コマンドがエラー(終了コード 0以外)を返したら即座にスクリプトを停止
#-u (nounset): 未定義の変数を参照した場合にエラーを出してスクリプトを停止
#-o pipefail: パイプライン内のいずれかのコマンドが失敗した場合にパイプライン全体を失敗とみなす
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
# -z: 文字列が空かどうかをチェックする条件式（空ならtrueを返す）
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ Error: OPENAI_API_KEY environment variable is not set"
  exit 1
fi

if [ -z "${AI_MODEL:-}" ]; then
  echo "❌ Error: AI_MODEL environment variable is not set"
  exit 1
fi

if [ -z "${MAX_TOKENS:-}" ]; then
  echo "❌ Error: MAX_TOKENS environment variable is not set"
  exit 1
fi

if [ -z "${TEMPERATURE:-}" ]; then
  echo "❌ Error: TEMPERATURE environment variable is not set"
  exit 1
fi

if [ -z "${GITHUB_OUTPUT:-}" ]; then
  echo "❌ Error: GITHUB_OUTPUT environment variable is not set"
  exit 1
fi

# 差分ファイル引数の検証
# $#: スクリプトに渡された引数の数
# -lt: "less than" (未満)を意味する比較演算子
if [ $# -lt 1 ]; then
  echo "❌ エラー: 第1引数として差分ファイルのパスが必要です"
  exit 1
fi

# 差分ファイルのパスを取得
# $1: スクリプトに渡された最初の引数
DIFF_FILE="$1"

# 差分ファイルの存在確認
# -f: 指定されたパスが存在し、通常のファイルであるかどうかをチェックする条件式
if [ ! -f "$DIFF_FILE" ]; then
  echo "❌ Error: Diff file not found: $DIFF_FILE"
  exit 1
fi

# プロンプトファイルの作成
# mktempは一時ファイルを作成し、そのパスを返す (例: /tmp/tmp.abcd1234)
# このファイルはスクリプト終了時に自動的にクリーンアップされる (下記のtrapを参照)
TEMP_PROMPT=$(mktemp)
# trap: スクリプト終了時(EXITイベント)に'rm -f "$TEMP_PROMPT"'を実行して一時ファイルを削除
# これによりスクリプトが失敗または中断された場合でもクリーンアップが保証される
trap 'rm -f "$TEMP_PROMPT"' EXIT

# 指定されたプロンプトまたはデフォルトを使用
# -n: 文字列が空でないかをチェックする条件式(空だとfalseを返す)
# ${REVIEW_PROMPT:-} はREVIEW_PROMPTが設定されているかチェック:
# - REVIEW_PROMPTが設定されていて空でない場合: その値を使用
# - REVIEW_PROMPTが未設定または空の場合: "" (空文字列)を使用
# 2番目の[ "$REVIEW_PROMPT" != '' ]は結果が空でないかチェック
# これにより環境変数からカスタムプロンプトを使用するか、デフォルトにフォールバックできる
if [ -n "${REVIEW_PROMPT:-}" ] && [ "$REVIEW_PROMPT" != '' ]; then
  # カスタムプロンプトが提供されている場合は、それを直接使用
  echo "$REVIEW_PROMPT" > "$TEMP_PROMPT"
else
  # それ以外の場合は、デフォルトのレビュープロンプトテンプレートを使用
  # cat >: 標準標準入力をファイルに書き込む
  # << 'EOF' : ヒアドキュメントの開始。EOFまで続く全テキストを標準入力として扱う
  # 動作:この行の次から EOF という行が現れるまでの全テキストが、そのまま $TEMP_PROMPT ファイルに書き込まれます。
  cat > "$TEMP_PROMPT" << 'EOF'
You are an experienced software engineer. Please review the following code diff in detail and analyze it from the following perspectives in English:

1. Code Quality: Readability, maintainability, performance
2. Security: Potential vulnerabilities and security risks
3. Best Practices: Language and framework recommendations
4. Bug Potential: Logic errors and exception handling issues
5. Improvement Suggestions: Specific improvement proposals and refactoring suggestions

Output Format:
- Point out issues specifically and include relevant line numbers
- Provide implementable concrete examples for improvement suggestions
- Clearly indicate importance level (High, Medium, Low)

Code Diff:
EOF
fi

# 実際の差分内容をプロンプトに追加
# echo "": 空文字列(実際には改行1つ)を出力
# >>: 追記リダイレクト演算子でファイル末尾に追加
echo "" >> "$TEMP_PROMPT"
cat "$DIFF_FILE" >> "$TEMP_PROMPT"

# APIリクエストペイロードの作成
# OpenAI APIに送信するJSONペイロード用の別の一時ファイルを作成
TEMP_PAYLOAD=$(mktemp)
# trap: 複数の一時ファイルを処理するように更新
# 複数のtrapコマンドはクリーンアップリストに追加される(前のものを上書きしない)
trap 'rm -f "$TEMP_PROMPT" "$TEMP_PAYLOAD"' EXIT


# jqを使用してOpenAI Chat Completions API用のJSONペイロードを生成
# jq -n: 標準入力を読まず、フィルタ式から新しいJSONドキュメントを構築
# --arg name value: シェル変数を文字列としてjqに渡す (例: $model)
# --rawfile name file: ファイルの内容全体を文字列としてjqに渡す (例: $prompt)
# --argjson name value: シェル変数を数値/JSONとしてjqに渡す (例: $max_tokens, $temperature)
# > "$TEMP_PAYLOAD": 生成されたJSONを一時ファイルに書き込む
jq -n \
  --arg model "$AI_MODEL" \
  --rawfile prompt "$TEMP_PROMPT" \
  --argjson max_tokens "$MAX_TOKENS" \
  --argjson temperature "$TEMPERATURE" \
  '{
    model: $model,
    messages: [
      {
        role: "system",
        content: "You are a helpful and constructive code reviewer. Please provide detailed and practical feedback."
      },
      {
        role: "user",
        content: $prompt
      }
    ],
    max_tokens: $max_tokens,
    temperature: $temperature
  }' > "$TEMP_PAYLOAD"

echo "🔄 Sending request to OpenAI API (model: $AI_MODEL)..."

# APIリクエストの送信
# APIレスポンスを保存するための3つ目の一時ファイルを作成
API_RESPONSE=$(mktemp)
# trap: 3つすべての一時ファイルをクリーンアップするように更新
trap 'rm -f "$TEMP_PROMPT" "$TEMP_PAYLOAD" "$API_RESPONSE"' EXIT

# curlを使用してOpenAI APIにPOSTリクエストを送信し、HTTPステータスコードを取得
# curl -w "%{http_code}": レスポンス後にHTTPステータスコード(200, 400等)を標準出力に追加
# -s (silent): 進捗バーやエラーメッセージを非表示にする
# -X POST: HTTPメソッドとしてPOSTを指定
# -H "Authorization: Bearer ...": OpenAI API認証用のBearerトークンヘッダー
# -H "Content-Type: application/json": リクエストボディがJSON形式であることを宣言
# -d @"$TEMP_PAYLOAD": @でファイルからリクエストボディを読み込み(jqで生成したJSONペイロード)
# -o "$API_RESPONSE": レスポンスボディを指定したファイルに保存(標準出力には出さない)
# 結果: HTTPステータスコードだけが標準出力され、HTTP_CODE変数に格納される
HTTP_CODE=$(curl -w "%{http_code}" -s \
  -X POST \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d @"$TEMP_PAYLOAD" \
  -o "$API_RESPONSE" \
  https://api.openai.com/v1/chat/completions
)

# HTTPステータスのチェック
# -ne: "not equal" (等しくない)を意味する比較演算子
# jq -r: jqでJSONから生の文字列を抽出 (-rは引用符なしで出力)
if [ "$HTTP_CODE" -ne 200 ]; then
  echo "❌ エラー: OpenAI APIリクエストがHTTP $HTTP_CODEで失敗しました"
  ERROR_TYPE=$(jq -r '.error.type // "unknown"' "$API_RESPONSE" 2>/dev/null || echo "unknown")
  ERROR_MESSAGE=$(jq -r '.error.message // "API call failed"' "$API_RESPONSE" 2>/dev/null || echo "API call failed")
  echo "エラータイプ: $ERROR_TYPE"
  echo "エラーメッセージ: $ERROR_MESSAGE"
  exit 1
fi

# レビュー内容の抽出
# .choices[0].message.content: JSONパスでchoices配列の最初の要素のmessage.contentを取得jq -r: JSONから生の文字列を抽出 (-rは引用符なしで出力)
# .choices[0].message.content: OpenAI APIレスポンスからAI生成テキストを取得するJSONパス
# // "...": Alternative operator - 左側がnull/存在しない場合は右側のデフォルト値を使用
# 2>/dev/null: jqのエラー出力(stderr)を破棄してクリーンな出力を維持
# 
# 代替記述例:
# 1. select使用: jq -r 'select(.choices) | .choices[0].message.content'
# 2. try-catch: jq -r 'try .choices[0].message.content catch "No content"'
# 3. has条件: jq -r 'if has("choices") then .choices[0].message.content else "No content" end'
REVIEW_CONTENT=$(jq -r '.choices[0].message.content // "No review content received"' "$API_RESPONSE" 2>/dev/null)

if [ "$REVIEW_CONTENT" = "null" ] || [ "$REVIEW_CONTENT" = "No review content received" ] || [ -z "$REVIEW_CONTENT" ]; then
  echo "❌ Error: No valid review content received from OpenAI"
  API_ERROR=$(jq -r '.error.message // "Unknown API error"' "$API_RESPONSE" 2>/dev/null || echo "Response parsing failed")
  echo "API indicated: $API_ERROR"
  exit 1
fi

# GitHub Actionsの出力に書き込み
{
  echo "review<<REVIEW_EOF"
  echo "$REVIEW_CONTENT"
  echo "REVIEW_EOF"
} >> "$GITHUB_OUTPUT"

# アーティファクトアップロード用にファイルにも保存
echo "$REVIEW_CONTENT" > tmp/ai_review_output.md

echo "✅ AI review generated successfully"
