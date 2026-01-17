# gh issue - Issue 操作コマンド

`gh issue` コマンドで Issue の作成、表示、編集、クローズなどを行います。

---

## 目次

- [gh issue create - 作成](#gh-issue-create---作成)
- [gh issue list - 一覧](#gh-issue-list---一覧)
- [gh issue view - 表示](#gh-issue-view---表示)
- [gh issue status - ステータス](#gh-issue-status---ステータス)
- [gh issue close - クローズ](#gh-issue-close---クローズ)
- [gh issue reopen - 再オープン](#gh-issue-reopen---再オープン)
- [gh issue edit - 編集](#gh-issue-edit---編集)
- [gh issue comment - コメント](#gh-issue-comment---コメント)
- [gh issue delete - 削除](#gh-issue-delete---削除)
- [gh issue transfer - 移動](#gh-issue-transfer---移動)
- [gh issue pin/unpin - ピン留め](#gh-issue-pinunpin---ピン留め)
- [gh issue develop - ブランチ作成](#gh-issue-develop---ブランチ作成)

---

## gh issue create - 作成

新しい Issue を作成します。

### 基本構文

```bash
gh issue create [flags]
```

### 使用例

```bash
# 対話形式で作成
gh issue create

# タイトルと本文を指定
gh issue create --title "バグ報告" --body "ここにバグの詳細"

# ラベルを付けて作成
gh issue create --title "機能追加" --label "enhancement"

# 複数ラベルを付ける
gh issue create --title "緊急バグ" --label "bug" --label "priority:high"

# 担当者を指定
gh issue create --title "タスク" --assignee "@me"
gh issue create --title "タスク" --assignee "username"

# マイルストーンを指定
gh issue create --title "v1.0対応" --milestone "v1.0"

# プロジェクトに追加
gh issue create --title "タスク" --project "My Project"

# ブラウザで作成画面を開く
gh issue create --web

# テンプレートを使用
gh issue create --template bug_report.md

# 別リポジトリに作成
gh issue create --repo owner/repo --title "Issue title"

# 本文をファイルから読み込み
gh issue create --title "タスク" --body-file ./issue_body.md
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-t, --title <string>` | タイトル |
| `-b, --body <string>` | 本文 |
| `-F, --body-file <file>` | 本文をファイルから読み込み |
| `-l, --label <strings>` | ラベル |
| `-a, --assignee <users>` | 担当者（@me で自分） |
| `-m, --milestone <name>` | マイルストーン |
| `-p, --project <name>` | プロジェクト |
| `-w, --web` | ブラウザで開く |
| `--template <file>` | Issue テンプレート |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### 本文の Markdown 例

```bash
gh issue create --title "機能リクエスト" --body "## 概要
新機能の説明

## 詳細
- 要件1
- 要件2

## 期待される動作
ユーザーが〇〇できるようになる"
```

---

## gh issue list - 一覧

Issue の一覧を表示します。

### 基本構文

```bash
gh issue list [flags]
```

### 使用例

```bash
# オープンな Issue 一覧（デフォルト）
gh issue list

# クローズ済み Issue
gh issue list --state closed

# すべての Issue
gh issue list --state all

# 自分に割り当てられた Issue
gh issue list --assignee "@me"

# 特定ユーザーに割り当てられた Issue
gh issue list --assignee "username"

# ラベルでフィルタ
gh issue list --label "bug"
gh issue list --label "bug" --label "priority:high"

# 作成者でフィルタ
gh issue list --author "@me"

# マイルストーンでフィルタ
gh issue list --milestone "v1.0"

# 表示件数を指定
gh issue list --limit 100

# 検索クエリを使用
gh issue list --search "is:open label:bug"

# JSON 形式で出力
gh issue list --json number,title,state

# テーブル形式で表示（デフォルト）
gh issue list

# 別リポジトリの Issue
gh issue list --repo owner/repo
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-s, --state <string>` | open/closed/all（デフォルト: open） |
| `-a, --assignee <user>` | 担当者でフィルタ |
| `-A, --author <user>` | 作成者でフィルタ |
| `-l, --label <strings>` | ラベルでフィルタ |
| `-m, --milestone <name>` | マイルストーンでフィルタ |
| `-S, --search <query>` | 検索クエリ |
| `-L, --limit <int>` | 表示件数（デフォルト: 30） |
| `--json <fields>` | JSON 形式で出力 |
| `-q, --jq <expression>` | jq 式でフィルタ |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

### 検索クエリ例

```bash
# 高優先度のバグ
gh issue list --search "is:open label:bug label:priority:high"

# 1週間以内に作成された Issue
gh issue list --search "is:open created:>=$(date -v-7d +%Y-%m-%d)"

# 特定キーワードを含む
gh issue list --search "is:open authentication"

# コメントがない Issue
gh issue list --search "is:open comments:0"
```

### JSON 出力でスクリプト連携

```bash
# Issue 番号のリストを取得
gh issue list --json number -q '.[].number'

# タイトルと URL を取得
gh issue list --json number,title,url --jq '.[] | "\(.number): \(.title)"'
```

---

## gh issue view - 表示

Issue の詳細を表示します。

### 基本構文

```bash
gh issue view <number> [flags]
```

### 使用例

```bash
# Issue の詳細を表示
gh issue view 123

# ブラウザで開く
gh issue view 123 --web

# コメントも表示
gh issue view 123 --comments

# JSON 形式で出力
gh issue view 123 --json title,body,state,labels

# 別リポジトリの Issue
gh issue view 123 --repo owner/repo
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-w, --web` | ブラウザで開く |
| `-c, --comments` | コメントを表示 |
| `--json <fields>` | JSON 形式で出力 |
| `-q, --jq <expression>` | jq 式でフィルタ |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh issue status - ステータス

自分に関連する Issue のステータスを表示します。

### 基本構文

```bash
gh issue status [flags]
```

### 使用例

```bash
# 関連 Issue の概要を表示
gh issue status
```

### 出力例

```
Issues assigned to you
  #123 [Bug] アプリがクラッシュする  (bug, priority:high)

Issues mentioning you
  #456 機能追加の提案  (enhancement)

Issues opened by you
  #789 ドキュメント更新  (documentation)
```

---

## gh issue close - クローズ

Issue をクローズします。

### 基本構文

```bash
gh issue close <number> [flags]
```

### 使用例

```bash
# Issue をクローズ
gh issue close 123

# コメント付きでクローズ
gh issue close 123 --comment "対応完了しました"

# 理由を指定してクローズ
gh issue close 123 --reason "completed"
gh issue close 123 --reason "not planned"
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-c, --comment <string>` | クローズ時のコメント |
| `-r, --reason <string>` | 理由（completed/not planned） |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh issue reopen - 再オープン

クローズした Issue を再オープンします。

### 基本構文

```bash
gh issue reopen <number> [flags]
```

### 使用例

```bash
# Issue を再オープン
gh issue reopen 123

# コメント付きで再オープン
gh issue reopen 123 --comment "再調査が必要になりました"
```

---

## gh issue edit - 編集

Issue を編集します。

### 基本構文

```bash
gh issue edit <number> [flags]
```

### 使用例

```bash
# タイトルを変更
gh issue edit 123 --title "新しいタイトル"

# 本文を変更
gh issue edit 123 --body "新しい本文"

# ラベルを追加
gh issue edit 123 --add-label "in-progress"

# ラベルを削除
gh issue edit 123 --remove-label "waiting"

# 担当者を追加
gh issue edit 123 --add-assignee "username"

# 担当者を削除
gh issue edit 123 --remove-assignee "old-user"

# マイルストーンを設定
gh issue edit 123 --milestone "v2.0"

# マイルストーンをクリア
gh issue edit 123 --milestone ""

# 複数の変更を同時に
gh issue edit 123 \
  --title "更新されたタイトル" \
  --add-label "reviewed" \
  --remove-label "needs-review" \
  --add-assignee "@me"
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-t, --title <string>` | タイトル |
| `-b, --body <string>` | 本文 |
| `-F, --body-file <file>` | 本文をファイルから |
| `--add-label <strings>` | ラベルを追加 |
| `--remove-label <strings>` | ラベルを削除 |
| `--add-assignee <users>` | 担当者を追加 |
| `--remove-assignee <users>` | 担当者を削除 |
| `-m, --milestone <name>` | マイルストーン |
| `--add-project <names>` | プロジェクトに追加 |
| `--remove-project <names>` | プロジェクトから削除 |
| `-R, --repo <owner/repo>` | 対象リポジトリ |

---

## gh issue comment - コメント

Issue にコメントを追加します。

### 基本構文

```bash
gh issue comment <number> [flags]
```

### 使用例

```bash
# 対話形式でコメント
gh issue comment 123

# コメントを直接指定
gh issue comment 123 --body "コメント内容"

# ファイルからコメント
gh issue comment 123 --body-file ./comment.md

# エディタでコメントを編集
gh issue comment 123 --editor

# ブラウザで開く
gh issue comment 123 --web

# 最後のコメントを編集
gh issue comment 123 --edit-last

# 最後のコメントを削除
gh issue comment 123 --delete-last
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-b, --body <string>` | コメント内容 |
| `-F, --body-file <file>` | ファイルから読み込み |
| `-e, --editor` | エディタで編集 |
| `-w, --web` | ブラウザで開く |
| `--edit-last` | 最後のコメントを編集 |

---

## gh issue delete - 削除

Issue を削除します。

### 基本構文

```bash
gh issue delete <number> [flags]
```

### 使用例

```bash
# Issue を削除
gh issue delete 123

# 確認をスキップ
gh issue delete 123 --yes
```

> ⚠️ **注意**: 削除は取り消せません。通常はクローズを使用することを推奨します。

---

## gh issue transfer - 移動

Issue を別のリポジトリに移動します。

### 基本構文

```bash
gh issue transfer <number> <destination-repo> [flags]
```

### 使用例

```bash
# Issue を別リポジトリに移動
gh issue transfer 123 owner/other-repo
```

---

## gh issue pin/unpin - ピン留め

Issue をピン留め/解除します。

### 使用例

```bash
# Issue をピン留め
gh issue pin 123

# ピン留め解除
gh issue unpin 123
```

---

## gh issue develop - ブランチ作成

Issue に関連するブランチを作成します。

### 基本構文

```bash
gh issue develop <number> [flags]
```

### 使用例

```bash
# Issue からブランチを作成
gh issue develop 123

# ブランチ名を指定
gh issue develop 123 --name fix-bug-123

# ベースブランチを指定
gh issue develop 123 --base develop

# ブランチ作成後にチェックアウト
gh issue develop 123 --checkout
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `-n, --name <string>` | ブランチ名 |
| `-b, --base <string>` | ベースブランチ |
| `-c, --checkout` | 作成後にチェックアウト |

---

## 実践的なワークフロー

### バグ報告から修正まで

```bash
# 1. バグ Issue を作成
gh issue create \
  --title "ログイン時にエラーが発生" \
  --body "## 再現手順
1. ログインページにアクセス
2. 正しい認証情報を入力
3. エラーが表示される

## 期待される動作
正常にログインできる

## 環境
- OS: macOS 14.0
- Browser: Chrome 120" \
  --label "bug" \
  --assignee "@me"

# 2. ブランチを作成
gh issue develop 123 --checkout --name fix/login-error

# 3. 修正してコミット
git add .
git commit -m "Fix login error (#123)"

# 4. PR を作成（Issue をリンク）
gh pr create --title "Fix login error" --body "Closes #123"
```

### Issue の一括処理

```bash
# 古い Issue をクローズ
gh issue list --search "is:open created:<2024-01-01" --json number -q '.[].number' | \
  xargs -I {} gh issue close {} --reason "not planned" --comment "古い Issue のため自動クローズ"

# 特定ラベルの Issue を担当者に割り当て
gh issue list --label "needs-triage" --json number -q '.[].number' | \
  xargs -I {} gh issue edit {} --add-assignee "triage-team"
```

### Issue テンプレートの活用

```bash
# テンプレート一覧を確認してから作成
gh issue create --web

# 特定テンプレートを使用
gh issue create --template bug_report.md
gh issue create --template feature_request.md
```

---

## 参考

- [gh issue 公式マニュアル](https://cli.github.com/manual/gh_issue)
