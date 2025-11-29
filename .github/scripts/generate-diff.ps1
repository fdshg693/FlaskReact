#!/usr/bin/env pwsh
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

$ErrorActionPreference = 'Stop'

# 注記: PowerShellの用語:
# - $ErrorActionPreference = 'Stop': エラーが発生した場合にスクリプトを停止(Bashのset -eに相当)
# 
# - Test-Path: ファイルやディレクトリの存在を確認するコマンドレット
#              例: Test-Path $env:GITHUB_OUTPUT
# 
# - $env:VARIABLE: 環境変数にアクセスする構文
#              例: $env:GITHUB_OUTPUT, $env:PR_BASE_REF
#
# - [string]::IsNullOrEmpty(): 文字列がnullまたは空かをチェック
#              例: -not [string]::IsNullOrEmpty($env:PR_BASE_REF)
#
# - Get-Item: ファイルやディレクトリの情報を取得
#              .Length プロパティでファイルサイズを取得(Bashの-sテストに相当)

# 必須環境変数の検証
if ([string]::IsNullOrEmpty($env:GITHUB_OUTPUT)) {
    Write-Host "❌ Error: GITHUB_OUTPUT environment variable is not set" -ForegroundColor Red
    exit 1
}

# 比較用のベースブランチを決定
$PR_BASE_REF = $env:PR_BASE_REF
$INPUT_TARGET = $env:INPUT_TARGET

if (-not [string]::IsNullOrEmpty($PR_BASE_REF) -and $PR_BASE_REF -ne 'null') {
    $BASE_BRANCH = $PR_BASE_REF
}
elseif (-not [string]::IsNullOrEmpty($INPUT_TARGET) -and $INPUT_TARGET -ne 'null') {
    $BASE_BRANCH = $INPUT_TARGET
}
else {
    $BASE_BRANCH = 'main'
}

Write-Host "Comparing against base branch: $BASE_BRANCH"

# ベースブランチをフェッチ
# git branch -r: リモート追跡ブランチの一覧を表示
try {
    git fetch origin $BASE_BRANCH
}
catch {
    Write-Host "❌ エラー: ベースブランチ'$BASE_BRANCH'のフェッチに失敗しました" -ForegroundColor Red
    Write-Host "利用可能なブランチ:"
    git branch -r
    exit 1
}

# tmpディレクトリが存在しない場合は作成
if (-not (Test-Path -Path "tmp")) {
    New-Item -ItemType Directory -Path "tmp" | Out-Null
}

# 差分を生成
# git diff "A...B": 3つのドット構文 - AとBの共通祖先(マージベース)からBまでの変更を表示
#                   プルリクエストで実際に追加された変更のみを抽出するのに最適
#                   (2つのドット"A..B"は単純な差分比較で、ベースブランチの変更も含まれる)
# --unified=3: 差分の前後3行のコンテキストを含める(デフォルト値、明示的に指定)
# --no-color: ANSIカラーコードを出力しない(ファイル保存やAPI送信用にプレーンテキスト化)
# --ignore-space-change: 空白のみの変更を無視(インデント調整などのノイズを除外)
# > tmp/diff.patch: 出力をファイルにリダイレクト
# 注記: PowerShellの文字化け対策として以下を実施:
# 1. [Console]::OutputEncoding をUTF-8に設定して git diff の出力を正しく受け取る
# 2. Out-File -Encoding utf8NoBOM でBOMなしUTF-8で保存(PS 6.0+)
#    または [System.IO.File]::WriteAllText で明示的にUTF-8エンコーディングを指定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$diffContent = git diff "origin/$BASE_BRANCH...HEAD" `
    --unified=3 `
    --no-color `
    --ignore-space-change

# UTF-8 (BOMなし) でファイルに保存
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Resolve-Path ".\tmp\diff.patch").Path, $diffContent, $utf8NoBom)

# 差分に内容があるかチェック
if (-not (Test-Path -Path "tmp/diff.patch") -or (Get-Item "tmp/diff.patch").Length -eq 0) {
    Write-Host "ℹ️ 現在のブランチと$BASE_BRANCHの間に変更が検出されませんでした" -ForegroundColor Yellow
    Add-Content -Path $env:GITHUB_OUTPUT -Value "has_changes=false"
    exit 0
}

$lineCount = (Get-Content "tmp/diff.patch" | Measure-Object -Line).Lines
Add-Content -Path $env:GITHUB_OUTPUT -Value "has_changes=true"
Write-Host "✅ Generated diff file ($lineCount lines)" -ForegroundColor Green
