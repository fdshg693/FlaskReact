# AI コードレビュー生成スクリプト (PowerShell版)
# このスクリプトは差分を生成し、OpenAI API を使用して AI コードレビューを作成します
#
# 使用方法:
#   .\scripts\generate_pr.ps1 [base_branch]
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

# エラー時にスクリプトを停止
$ErrorActionPreference = "Stop"

# スクリプトのディレクトリを取得（絶対パス）
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# プロジェクトルートに移動
Set-Location $ProjectRoot

Write-Host "🚀 AI コードレビュープロセスを開始します" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# .env ファイルから環境変数を読み込み
$EnvFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "❌ エラー: プロジェクトルートに .env ファイルが見つかりません" -ForegroundColor Red
    Write-Host "sample.env を参考に .env ファイルを作成してください" -ForegroundColor Yellow
    exit 1
}

Write-Host "📁 .env から環境変数を読み込み中..." -ForegroundColor Green

# .env ファイルを読み込んで環境変数に設定
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    # コメント行と空行をスキップ
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            # 引用符を削除
            $value = $value -replace '^["'']|["'']$', ''
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# 必須の環境変数を検証
$OpenAIKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "Process")
if ([string]::IsNullOrEmpty($OpenAIKey)) {
    Write-Host "❌ エラー: OPENAI_API_KEY が .env ファイルに設定されていません" -ForegroundColor Red
    exit 1
}

# AI モデル設定のデフォルト値を設定
$AIModel = [Environment]::GetEnvironmentVariable("AI_MODEL", "Process")
if ([string]::IsNullOrEmpty($AIModel)) {
    $AIModel = "gpt-4.1"
    [Environment]::SetEnvironmentVariable("AI_MODEL", $AIModel, "Process")
}

$MaxTokens = [Environment]::GetEnvironmentVariable("MAX_TOKENS", "Process")
if ([string]::IsNullOrEmpty($MaxTokens)) {
    $MaxTokens = "10000"
    [Environment]::SetEnvironmentVariable("MAX_TOKENS", $MaxTokens, "Process")
}

$Temperature = [Environment]::GetEnvironmentVariable("TEMPERATURE", "Process")
if ([string]::IsNullOrEmpty($Temperature)) {
    $Temperature = "0.1"
    [Environment]::SetEnvironmentVariable("TEMPERATURE", $Temperature, "Process")
}

Write-Host "✅ 環境変数の読み込み完了" -ForegroundColor Green
Write-Host "   - モデル: $AIModel"
Write-Host "   - 最大トークン数: $MaxTokens"
Write-Host "   - Temperature: $Temperature"
Write-Host ""

# ベースブランチを決定
$BaseBranch = if ($args.Count -gt 0) { $args[0] } else { "main" }
Write-Host "📊 比較対象のベースブランチ: $BaseBranch" -ForegroundColor Cyan

# 一時的な GITHUB_OUTPUT ファイルを作成
$TempOutput = New-TemporaryFile
[Environment]::SetEnvironmentVariable("GITHUB_OUTPUT", $TempOutput.FullName, "Process")

try {
    # ステップ 1: generate-diff.ps1 を使用して差分を生成
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "📝 ステップ 1: 差分を生成中..." -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan

    # generate-diff.ps1 に必要な変数を設定
    [Environment]::SetEnvironmentVariable("INPUT_TARGET", $BaseBranch, "Process")

    # generate-diff スクリプトを実行
    $DiffScript = Join-Path $ProjectRoot ".github\scripts\generate-diff.ps1"
    if (-not (Test-Path $DiffScript)) {
        Write-Host "❌ エラー: .github\scripts\generate-diff.ps1 が見つかりません" -ForegroundColor Red
        exit 1
    }

    & pwsh -File $DiffScript

    # 差分が生成されたか確認
    $DiffFile = Join-Path $ProjectRoot "tmp\diff.patch"
    if (-not (Test-Path $DiffFile)) {
        Write-Host "❌ エラー: tmp\diff.patch が作成されませんでした" -ForegroundColor Red
        exit 1
    }

    # 変更があるか確認
    $OutputContent = Get-Content $TempOutput.FullName -Raw
    $HasChanges = $false
    if ($OutputContent -match "has_changes=(\w+)") {
        $HasChanges = $matches[1] -eq "true"
    }

    if (-not $HasChanges) {
        Write-Host "ℹ️ 変更が検出されませんでした。AI レビューをスキップします。" -ForegroundColor Yellow
        exit 0
    }

    Write-Host "✅ 差分の生成が完了しました" -ForegroundColor Green
    Write-Host ""

    # ステップ 2: generate-ai-review.ps1 を使用して AI レビューを生成
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "🤖 ステップ 2: AI レビューを生成中..." -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan

    # AI レビュースクリプトを実行
    $ReviewScript = Join-Path $ProjectRoot ".github\scripts\generate-ai-review.ps1"
    if (-not (Test-Path $ReviewScript)) {
        Write-Host "❌ エラー: .github\scripts\generate-ai-review.ps1 が見つかりません" -ForegroundColor Red
        exit 1
    }

    & pwsh -File $ReviewScript -DiffFile $DiffFile

    # レビューが生成されたか確認
    $ReviewFile = Join-Path $ProjectRoot "tmp\ai_review_output.md"
    if (-not (Test-Path $ReviewFile)) {
        Write-Host "❌ エラー: tmp\ai_review_output.md が作成されませんでした" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "✅ AI コードレビューが完了しました！" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📄 生成されたファイル:"
    Write-Host "   - tmp\diff.patch: $BaseBranch と現在のブランチ間の Git 差分"
    Write-Host "   - tmp\ai_review_output.md: AI が生成したコードレビュー"
    Write-Host ""
    Write-Host "📖 レビューを確認するには:"
    Write-Host "   Get-Content tmp\ai_review_output.md"
    Write-Host ""
}
finally {
    # 一時ファイルのクリーンアップ
    if (Test-Path $TempOutput.FullName) {
        Remove-Item $TempOutput.FullName -Force -ErrorAction SilentlyContinue
    }
}
