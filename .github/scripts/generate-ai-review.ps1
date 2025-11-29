# AIコードレビュー生成スクリプト (PowerShell版)
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
#   -DiffFile: 差分ファイルのパス
#
# 使用例:
#   .\generate-ai-review.ps1 -DiffFile "path\to\diff.txt"

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DiffFile
)

# エラー発生時にスクリプトを停止
$ErrorActionPreference = "Stop"

# 注記: PowerShellの用語:
# - [System.IO.Path]::GetTempFileName(): ユニークな名前の一時ファイルを作成し、そのファイルパスを返す。
# - try/finally: スクリプト終了時（正常終了またはエラー時）にクリーンアップを実行。
# - $env:VARIABLE: 環境変数へのアクセス方法。

# 一時ファイルのリスト（クリーンアップ用）
$tempFiles = @()

function Cleanup-TempFiles {
    foreach ($file in $script:tempFiles) {
        if (Test-Path $file) {
            Remove-Item $file -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    # 必須環境変数の検証
    if ([string]::IsNullOrEmpty($env:OPENAI_API_KEY)) {
        Write-Error "❌ Error: OPENAI_API_KEY environment variable is not set"
        exit 1
    }

    if ([string]::IsNullOrEmpty($env:AI_MODEL)) {
        Write-Error "❌ Error: AI_MODEL environment variable is not set"
        exit 1
    }

    if ([string]::IsNullOrEmpty($env:MAX_TOKENS)) {
        Write-Error "❌ Error: MAX_TOKENS environment variable is not set"
        exit 1
    }

    if ([string]::IsNullOrEmpty($env:TEMPERATURE)) {
        Write-Error "❌ Error: TEMPERATURE environment variable is not set"
        exit 1
    }

    if ([string]::IsNullOrEmpty($env:GITHUB_OUTPUT)) {
        Write-Error "❌ Error: GITHUB_OUTPUT environment variable is not set"
        exit 1
    }

    # 差分ファイルの存在確認
    if (-not (Test-Path $DiffFile)) {
        Write-Error "❌ Error: Diff file not found: $DiffFile"
        exit 1
    }

    # プロンプトファイルの作成
    $tempPrompt = [System.IO.Path]::GetTempFileName()
    $tempFiles += $tempPrompt

    # 指定されたプロンプトまたはデフォルトを使用
    if (-not [string]::IsNullOrEmpty($env:REVIEW_PROMPT)) {
        # カスタムプロンプトが提供されている場合は、それを直接使用
        $promptContent = $env:REVIEW_PROMPT
    }
    else {
        # それ以外の場合は、デフォルトのレビュープロンプトテンプレートを使用
        $promptContent = @"
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
"@
    }

    # 実際の差分内容をプロンプトに追加
    $diffContent = Get-Content -Path $DiffFile -Raw -Encoding UTF8
    $fullPrompt = $promptContent + "`n" + $diffContent
    Set-Content -Path $tempPrompt -Value $fullPrompt -Encoding UTF8

    # APIリクエストペイロードの作成
    $maxTokens = [int]$env:MAX_TOKENS
    $temperature = [double]$env:TEMPERATURE

    $payload = @{
        model = $env:AI_MODEL
        messages = @(
            @{
                role = "system"
                content = "You are a helpful and constructive code reviewer. Please provide detailed and practical feedback."
            },
            @{
                role = "user"
                content = $fullPrompt
            }
        )
        max_tokens = $maxTokens
        temperature = $temperature
    } | ConvertTo-Json -Depth 10 -Compress

    Write-Host "🔄 Sending request to OpenAI API (model: $($env:AI_MODEL))..."

    # APIリクエストの送信
    $headers = @{
        "Authorization" = "Bearer $($env:OPENAI_API_KEY)"
        "Content-Type" = "application/json"
    }

    try {
        $response = Invoke-RestMethod -Uri "https://api.openai.com/v1/chat/completions" `
            -Method Post `
            -Headers $headers `
            -Body $payload `
            -ContentType "application/json; charset=utf-8"
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Error "❌ Error: OpenAI API request failed with HTTP $statusCode"
        
        # エラーレスポンスの詳細を取得
        $errorResponse = $_.ErrorDetails.Message
        if ($errorResponse) {
            try {
                $errorJson = $errorResponse | ConvertFrom-Json
                $errorType = if ($errorJson.error.type) { $errorJson.error.type } else { "unknown" }
                $errorMessage = if ($errorJson.error.message) { $errorJson.error.message } else { "API call failed" }
                Write-Host "Error Type: $errorType"
                Write-Host "Error Message: $errorMessage"
            }
            catch {
                Write-Host "Error Details: $errorResponse"
            }
        }
        exit 1
    }

    # レビュー内容の抽出
    $reviewContent = $response.choices[0].message.content

    if ([string]::IsNullOrEmpty($reviewContent)) {
        Write-Error "❌ Error: No valid review content received from OpenAI"
        exit 1
    }

    # GitHub Actionsの出力に書き込み
    # ヒアドキュメント形式で複数行の出力を書き込む
    $githubOutput = @"
review<<REVIEW_EOF
$reviewContent
REVIEW_EOF
"@
    Add-Content -Path $env:GITHUB_OUTPUT -Value $githubOutput -Encoding UTF8

    # アーティファクトアップロード用にファイルにも保存
    $outputDir = "tmp"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    Set-Content -Path "$outputDir\ai_review_output.md" -Value $reviewContent -Encoding UTF8

    Write-Host "✅ AI review generated successfully"
}
finally {
    # 一時ファイルのクリーンアップ
    Cleanup-TempFiles
}
