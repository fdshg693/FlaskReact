# GitHub CLI (gh コマンド) ガイド

GitHub CLI (gh) は、GitHub をコマンドラインから操作するための公式ツールです。

## 目次

- [00_INDEX.md](./00_INDEX.md) - このファイル（概要・セットアップ）
- [01_リポジトリ操作.md](./01_リポジトリ操作.md) - gh repo コマンド
- [02_Issue操作.md](./02_Issue操作.md) - gh issue コマンド
- [03_PR操作.md](./03_PR操作.md) - gh pr コマンド
- [04_Actions_Workflow.md](./04_Actions_Workflow.md) - gh workflow / gh run コマンド
- [05_その他便利コマンド.md](./05_その他便利コマンド.md) - gist, release, search, api, alias 等

---

## GitHub CLI とは

GitHub CLI は、ターミナルから GitHub の機能を直接操作できるコマンドラインツールです。

### Git コマンドとの違い

| 項目 | Git コマンド | GitHub CLI (gh) |
|------|-------------|-----------------|
| 目的 | ローカル/リモートの Git リポジトリ操作 | GitHub 固有機能の操作 |
| 操作対象 | コミット、ブランチ、マージ等 | PR、Issue、Actions、リリース等 |
| 認証 | SSH キーや credential helper | GitHub アカウント認証 |

**Git コマンドの例:**
```bash
git clone https://github.com/owner/repo.git
git commit -m "message"
git push origin main
```

**GitHub CLI の例:**
```bash
gh repo clone owner/repo
gh pr create --title "feature" --body "説明"
gh issue list
```

---

## インストール

### macOS (Homebrew)

```bash
brew install gh
```

### Ubuntu / Debian

```bash
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

### Windows (winget)

```powershell
winget install --id GitHub.cli
```

### Windows (Scoop)

```powershell
scoop install gh
```

### バージョン確認

```bash
gh --version
```

---

## 認証設定

### 初期認証

```bash
gh auth login
```

対話形式で以下を選択:
1. **GitHub.com** または **GitHub Enterprise Server**
2. 認証方法: **HTTPS** または **SSH**
3. ブラウザまたはトークンで認証

### 認証状態の確認

```bash
gh auth status
```

出力例:
```
github.com
  ✓ Logged in to github.com account username (keyring)
  - Active account: true
  - Git operations protocol: https
  - Token: gho_****
  - Token scopes: 'gist', 'read:org', 'repo', 'workflow'
```

### 認証のリフレッシュ

```bash
gh auth refresh
```

### ログアウト

```bash
gh auth logout
```

### GitHub Enterprise Server の認証

```bash
gh auth login --hostname enterprise.example.com
```

---

## 基本設定

### 設定の確認

```bash
gh config list
```

### エディタの設定

```bash
# VS Code を設定
gh config set editor "code --wait"

# vim を設定
gh config set editor vim
```

### デフォルトプロトコルの設定

```bash
# HTTPS を使用
gh config set git_protocol https

# SSH を使用
gh config set git_protocol ssh
```

### ページャーの設定

```bash
gh config set pager less
```

### ブラウザの設定

```bash
gh config set browser firefox
```

---

## 環境変数

| 変数名 | 説明 |
|--------|------|
| `GITHUB_TOKEN` | 認証トークン |
| `GH_HOST` | デフォルトホスト |
| `GH_ENTERPRISE_TOKEN` | Enterprise Server 用トークン |
| `GH_EDITOR` | エディタ指定 |
| `GH_BROWSER` | ブラウザ指定 |
| `GH_PAGER` | ページャー指定 |
| `NO_COLOR` | カラー出力を無効化 |

---

## コマンド一覧

### コアコマンド

| コマンド | 説明 |
|---------|------|
| `gh auth` | 認証管理 |
| `gh browse` | ブラウザで開く |
| `gh codespace` | Codespace 管理 |
| `gh gist` | Gist 管理 |
| `gh issue` | Issue 管理 |
| `gh org` | Organization 管理 |
| `gh pr` | Pull Request 管理 |
| `gh project` | Project 管理 |
| `gh release` | リリース管理 |
| `gh repo` | リポジトリ管理 |

### GitHub Actions コマンド

| コマンド | 説明 |
|---------|------|
| `gh cache` | キャッシュ管理 |
| `gh run` | ワークフロー実行管理 |
| `gh workflow` | ワークフロー管理 |

### その他のコマンド

| コマンド | 説明 |
|---------|------|
| `gh alias` | エイリアス管理 |
| `gh api` | GitHub API 直接呼び出し |
| `gh config` | 設定管理 |
| `gh extension` | 拡張機能管理 |
| `gh label` | ラベル管理 |
| `gh search` | 検索 |
| `gh secret` | シークレット管理 |
| `gh ssh-key` | SSH キー管理 |
| `gh status` | ステータス確認 |
| `gh variable` | 変数管理 |

---

## ヘルプの使い方

```bash
# 全体ヘルプ
gh help

# コマンド別ヘルプ
gh repo --help
gh pr create --help

# マニュアルページ
gh help repo
```

---

## 参考リンク

- [GitHub CLI 公式サイト](https://cli.github.com/)
- [GitHub CLI マニュアル](https://cli.github.com/manual/)
- [GitHub CLI リポジトリ](https://github.com/cli/cli)
