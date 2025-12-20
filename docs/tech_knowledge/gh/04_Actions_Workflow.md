# gh workflow / gh run - GitHub Actions 操作コマンド

`gh workflow` と `gh run` コマンドで GitHub Actions のワークフローと実行を管理します。

---

## 目次

### gh workflow
- [gh workflow list - ワークフロー一覧](#gh-workflow-list---ワークフロー一覧)
- [gh workflow view - ワークフロー詳細](#gh-workflow-view---ワークフロー詳細)
- [gh workflow run - 手動実行](#gh-workflow-run---手動実行)
- [gh workflow enable/disable - 有効化/無効化](#gh-workflow-enabledisable---有効化無効化)

### gh run
- [gh run list - 実行履歴一覧](#gh-run-list---実行履歴一覧)
- [gh run view - 実行詳細](#gh-run-view---実行詳細)
- [gh run watch - リアルタイム監視](#gh-run-watch---リアルタイム監視)
- [gh run download - アーティファクト取得](#gh-run-download---アーティファクト取得)
- [gh run rerun - 再実行](#gh-run-rerun---再実行)
- [gh run cancel - キャンセル](#gh-run-cancel---キャンセル)

### gh cache
- [gh cache list - キャッシュ一覧](#gh-cache-list---キャッシュ一覧)
- [gh cache delete - キャッシュ削除](#gh-cache-delete---キャッシュ削除)

---

## gh workflow list - ワークフロー一覧

リポジトリのワークフロー一覧を表示します。

### 基本構文

```bash
gh workflow list [flags]
```

### 使用例

```bash
# 全ワークフロー一覧
gh workflow list

# 有効なワークフローのみ
gh workflow list --all

# 表示件数を指定
gh workflow list --limit 50

# JSON 形式で出力
gh workflow list --json id,name,state

# 別リポジトリのワークフロー
gh workflow list --repo owner/repo
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-a, --all` | 無効なものも含めて表示 |
| `-L, --limit <int>` | 表示件数 |
| `--json <fields>` | JSON 形式で出力 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### 出力例

```
NAME                   STATE   ID
CI                     active  12345678
Deploy                 active  12345679
Security Scan          disabled 12345680
```

---

## gh workflow view - ワークフロー詳細

ワークフローの詳細と最近の実行を表示します。

### 基本構文

```bash
gh workflow view [<workflow-id> | <workflow-name> | <filename>] [flags]
```

### 使用例

```bash
# 対話形式でワークフローを選択
gh workflow view

# ワークフロー ID で指定
gh workflow view 12345678

# ワークフロー名で指定
gh workflow view "CI"

# ファイル名で指定
gh workflow view ci.yml

# YAML ファイルの内容を表示
gh workflow view CI --yaml

# ブラウザで開く
gh workflow view CI --web

# 特定ブランチの実行を表示
gh workflow view CI --ref main

# JSON 形式で出力
gh workflow view CI --json id,name,path
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-y, --yaml` | YAML ファイル内容を表示 |
| `-w, --web` | ブラウザで開く |
| `-r, --ref <branch>` | ブランチ/タグを指定 |
| `--json <fields>` | JSON 形式で出力 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh workflow run - 手動実行

ワークフローを手動で実行します（workflow_dispatch イベント）。

### 基本構文

```bash
gh workflow run [<workflow-id> | <workflow-name> | <filename>] [flags]
```

### 使用例

```bash
# 対話形式で実行
gh workflow run

# ワークフロー名で実行
gh workflow run "Deploy"

# ファイル名で実行
gh workflow run deploy.yml

# ブランチを指定
gh workflow run deploy.yml --ref develop

# 入力パラメータを指定
gh workflow run deploy.yml -f environment=production

# 複数のパラメータ
gh workflow run deploy.yml \
  -f environment=staging \
  -f version=v1.2.3 \
  -f dry_run=false

# JSON ファイルからパラメータを読み込み
gh workflow run deploy.yml --json < params.json

# 別リポジトリで実行
gh workflow run deploy.yml --repo owner/repo
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-r, --ref <branch>` | ブランチ/タグを指定 |
| `-f, --field <key=value>` | 入力パラメータ |
| `-F, --raw-field <key=value>` | 生の入力パラメータ |
| `--json` | JSON で入力パラメータを渡す |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### workflow_dispatch の設定例

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      version:
        description: 'Version to deploy'
        required: true
      dry_run:
        description: 'Dry run mode'
        required: false
        default: 'true'
        type: boolean

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: |
          echo "Deploying ${{ inputs.version }} to ${{ inputs.environment }}"
          echo "Dry run: ${{ inputs.dry_run }}"
```

---

## gh workflow enable/disable - 有効化/無効化

ワークフローの有効化/無効化を切り替えます。

### 基本構文

```bash
gh workflow enable [<workflow-id> | <workflow-name> | <filename>] [flags]
gh workflow disable [<workflow-id> | <workflow-name> | <filename>] [flags]
```

### 使用例

```bash
# ワークフローを無効化
gh workflow disable "Security Scan"
gh workflow disable security-scan.yml

# ワークフローを有効化
gh workflow enable "Security Scan"
gh workflow enable security-scan.yml
```

---

## gh run list - 実行履歴一覧

ワークフローの実行履歴を表示します。

### 基本構文

```bash
gh run list [flags]
```

### 使用例

```bash
# 全実行履歴
gh run list

# 特定ワークフローの実行
gh run list --workflow "CI"
gh run list --workflow ci.yml

# ブランチでフィルタ
gh run list --branch main

# ステータスでフィルタ
gh run list --status success
gh run list --status failure
gh run list --status in_progress

# ユーザーでフィルタ
gh run list --user username

# イベントでフィルタ
gh run list --event push
gh run list --event pull_request

# コミット SHA でフィルタ
gh run list --commit abc1234

# 表示件数を指定
gh run list --limit 50

# JSON 形式で出力
gh run list --json databaseId,displayTitle,status,conclusion

# 別リポジトリの実行
gh run list --repo owner/repo
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-w, --workflow <string>` | ワークフローでフィルタ |
| `-b, --branch <string>` | ブランチでフィルタ |
| `-s, --status <string>` | ステータスでフィルタ |
| `-u, --user <string>` | ユーザーでフィルタ |
| `-e, --event <string>` | イベントでフィルタ |
| `-c, --commit <SHA>` | コミットでフィルタ |
| `-L, --limit <int>` | 表示件数 |
| `--json <fields>` | JSON 形式で出力 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### ステータス一覧

| ステータス | 説明 |
|-----------|------|
| `queued` | キュー待ち |
| `in_progress` | 実行中 |
| `completed` | 完了 |
| `waiting` | 待機中 |
| `requested` | リクエスト済み |
| `pending` | 保留中 |

### 結果（conclusion）一覧

| 結果 | 説明 |
|-----|------|
| `success` | 成功 |
| `failure` | 失敗 |
| `cancelled` | キャンセル |
| `skipped` | スキップ |
| `timed_out` | タイムアウト |
| `action_required` | 対応必要 |

### 出力例

```
STATUS  TITLE                           WORKFLOW  BRANCH  EVENT        ID          ELAPSED  AGE
✓       Fix login bug (#123)            CI        main    push         1234567890  2m34s    5m
✓       Add new feature (#122)          CI        main    push         1234567889  3m12s    1h
✗       Update dependencies             CI        deps    push         1234567888  1m45s    2h
*       Deploy to staging               Deploy    main    workflow_dispatch  1234567887  -        3h
```

---

## gh run view - 実行詳細

ワークフロー実行の詳細を表示します。

### 基本構文

```bash
gh run view [<run-id>] [flags]
```

### 使用例

```bash
# 対話形式で実行を選択
gh run view

# 実行 ID で詳細表示
gh run view 1234567890

# ジョブの詳細も表示
gh run view 1234567890 --verbose

# 特定ジョブのログを表示
gh run view 1234567890 --job 12345

# ログを表示
gh run view 1234567890 --log

# 失敗したジョブのログのみ
gh run view 1234567890 --log-failed

# ブラウザで開く
gh run view 1234567890 --web

# 終了コードで失敗/成功を判定
gh run view 1234567890 --exit-status

# JSON 形式で出力
gh run view 1234567890 --json jobs,status,conclusion
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-v, --verbose` | 詳細表示 |
| `-j, --job <id>` | 特定ジョブを表示 |
| `--log` | ログを表示 |
| `--log-failed` | 失敗ログのみ表示 |
| `-w, --web` | ブラウザで開く |
| `--exit-status` | 結果を終了コードで返す |
| `--json <fields>` | JSON 形式で出力 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### ジョブ情報の取得

```bash
# ジョブ一覧を取得
gh run view 1234567890 --json jobs -q '.jobs[] | "\(.name): \(.status) (\(.conclusion))"'

# 失敗ジョブのみ
gh run view 1234567890 --json jobs -q '.jobs[] | select(.conclusion == "failure")'
```

---

## gh run watch - リアルタイム監視

実行中のワークフローをリアルタイムで監視します。

### 基本構文

```bash
gh run watch [<run-id>] [flags]
```

### 使用例

```bash
# 対話形式で実行を選択
gh run watch

# 実行 ID で監視
gh run watch 1234567890

# 完了後に終了コードを返す
gh run watch 1234567890 --exit-status

# 更新間隔を指定（秒）
gh run watch 1234567890 --interval 5
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `--exit-status` | 結果を終了コードで返す |
| `-i, --interval <seconds>` | 更新間隔 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### CI 監視スクリプト

```bash
# プッシュ後、CI を監視してマージ
git push origin feature-branch
gh run watch --exit-status && gh pr merge --squash --delete-branch
```

---

## gh run download - アーティファクト取得

ワークフロー実行のアーティファクトをダウンロードします。

### 基本構文

```bash
gh run download [<run-id>] [flags]
```

### 使用例

```bash
# 対話形式でダウンロード
gh run download

# 実行 ID を指定
gh run download 1234567890

# 特定のアーティファクトのみ
gh run download 1234567890 --name "build-output"

# ダウンロード先を指定
gh run download 1234567890 --dir ./artifacts

# パターンでフィルタ
gh run download 1234567890 --pattern "*.zip"

# 最新の実行からダウンロード
gh run list --workflow "Build" --limit 1 --json databaseId -q '.[0].databaseId' | \
  xargs gh run download
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-n, --name <strings>` | アーティファクト名でフィルタ |
| `-p, --pattern <glob>` | パターンでフィルタ |
| `-D, --dir <path>` | ダウンロード先 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh run rerun - 再実行

ワークフロー実行を再実行します。

### 基本構文

```bash
gh run rerun [<run-id>] [flags]
```

### 使用例

```bash
# 実行全体を再実行
gh run rerun 1234567890

# 失敗したジョブのみ再実行
gh run rerun 1234567890 --failed

# 特定のジョブを再実行
gh run rerun 1234567890 --job build

# デバッグログを有効にして再実行
gh run rerun 1234567890 --debug
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `--failed` | 失敗ジョブのみ再実行 |
| `-j, --job <string>` | 特定ジョブを再実行 |
| `-d, --debug` | デバッグログを有効化 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh run cancel - キャンセル

実行中のワークフローをキャンセルします。

### 基本構文

```bash
gh run cancel [<run-id>] [flags]
```

### 使用例

```bash
# 実行をキャンセル
gh run cancel 1234567890

# 複数の実行をキャンセル
gh run list --status in_progress --json databaseId -q '.[].databaseId' | \
  xargs -I {} gh run cancel {}
```

---

## gh cache list - キャッシュ一覧

リポジトリのアクションキャッシュを一覧表示します。

### 基本構文

```bash
gh cache list [flags]
```

### 使用例

```bash
# キャッシュ一覧
gh cache list

# 表示件数を指定
gh cache list --limit 100

# ソート
gh cache list --sort created_at
gh cache list --sort last_accessed_at
gh cache list --sort size_in_bytes

# 降順/昇順
gh cache list --order desc
gh cache list --order asc

# キーでフィルタ
gh cache list --key "npm-"

# JSON 形式で出力
gh cache list --json id,key,sizeInBytes
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-L, --limit <int>` | 表示件数 |
| `-S, --sort <string>` | ソート基準 |
| `-O, --order <string>` | 並び順（asc/desc） |
| `-k, --key <string>` | キーでフィルタ |
| `--json <fields>` | JSON 形式で出力 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh cache delete - キャッシュ削除

アクションキャッシュを削除します。

### 基本構文

```bash
gh cache delete [<cache-id> | <cache-key> | --all] [flags]
```

### 使用例

```bash
# キャッシュ ID で削除
gh cache delete 12345

# キャッシュキーで削除
gh cache delete npm-linux-abc123

# 全キャッシュを削除
gh cache delete --all

# 確認をスキップ
gh cache delete --all --confirm
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-a, --all` | 全キャッシュを削除 |
| `--confirm` | 確認をスキップ |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## 実践的なワークフロー

### デプロイワークフロー

```bash
# 1. ワークフロー一覧を確認
gh workflow list

# 2. 手動デプロイ実行
gh workflow run deploy.yml \
  -f environment=production \
  -f version=$(git describe --tags --abbrev=0)

# 3. 実行を監視
gh run list --workflow deploy.yml --limit 1 --json databaseId -q '.[0].databaseId' | \
  xargs gh run watch --exit-status

# 4. ログを確認（失敗時）
gh run list --workflow deploy.yml --limit 1 --json databaseId -q '.[0].databaseId' | \
  xargs gh run view --log-failed
```

### CI デバッグ

```bash
# 1. 失敗した実行を確認
gh run list --status failure --limit 5

# 2. 詳細を確認
gh run view 1234567890 --verbose

# 3. ログを確認
gh run view 1234567890 --log-failed

# 4. デバッグモードで再実行
gh run rerun 1234567890 --debug

# 5. 監視
gh run watch 1234567890 --exit-status
```

### アーティファクト自動ダウンロード

```bash
#!/bin/bash
# 最新ビルドのアーティファクトをダウンロード

WORKFLOW="Build"
ARTIFACT_NAME="build-output"
DOWNLOAD_DIR="./downloads"

# 最新の成功した実行を取得
RUN_ID=$(gh run list --workflow "$WORKFLOW" --status completed --json databaseId,conclusion \
  -q '[.[] | select(.conclusion == "success")][0].databaseId')

if [ -n "$RUN_ID" ]; then
  echo "Downloading artifacts from run $RUN_ID..."
  gh run download "$RUN_ID" --name "$ARTIFACT_NAME" --dir "$DOWNLOAD_DIR"
  echo "Done!"
else
  echo "No successful run found"
  exit 1
fi
```

### キャッシュ管理

```bash
# キャッシュ使用状況を確認
gh cache list --json key,sizeInBytes -q 'map(.sizeInBytes) | add'

# 古いキャッシュを削除
gh cache list --sort last_accessed_at --order asc --limit 10 --json id -q '.[].id' | \
  xargs -I {} gh cache delete {}

# 特定プレフィックスのキャッシュを削除
gh cache list --key "npm-" --json id -q '.[].id' | \
  xargs -I {} gh cache delete {}
```

---

## 参考

- [gh workflow 公式マニュアル](https://cli.github.com/manual/gh_workflow)
- [gh run 公式マニュアル](https://cli.github.com/manual/gh_run)
- [gh cache 公式マニュアル](https://cli.github.com/manual/gh_cache)
