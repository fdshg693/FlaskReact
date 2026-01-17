#!/bin/bash

# Github Actionsを一括削除するスクリプト
# 使用方法:
#   ./clean_actions.sh              # 全てのワークフローを削除
#   ./clean_actions.sh workflow.yml # 特定のワークフローを削除

set -e

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ヘルプ表示
show_help() {
    cat << EOF
使用方法: $(basename "$0") [OPTIONS]

オプション:
    -w, --workflow WORKFLOW_NAME  特定のワークフローのみ削除 (例: workflow.yml)
    -a, --all                     全てのワークフローを削除 (デフォルト)
    -h, --help                    このヘルプメッセージを表示

例:
    $(basename "$0")                    # 全てのワークフローを削除
    $(basename "$0") -w deploy.yml      # deploy.ymlのみ削除
    $(basename "$0") --workflow test.yml # test.ymlのみ削除

EOF
}

# デフォルト値
WORKFLOW=""
DELETE_ALL=true

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workflow)
            WORKFLOW="$2"
            DELETE_ALL=false
            shift 2
            ;;
        -a|--all)
            DELETE_ALL=true
            WORKFLOW=""
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}エラー: 未知のオプション '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 確認プロンプト
confirm_deletion() {
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    read -p "本当に削除しますか? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "削除をキャンセルしました"
        exit 0
    fi
}

# 削除処理の実行
delete_runs() {
    local workflow_filter=""
    local description=""
    
    if [ -z "$WORKFLOW" ]; then
        description="全てのワークフロー実行"
    else
        workflow_filter="--workflow $WORKFLOW"
        description="ワークフロー '$WORKFLOW' の実行"
    fi
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Github Actions 削除ツール${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "削除対象: $description"
    echo ""
    
    # 削除対象のrun数を確認
    local count=$(gh run list $workflow_filter --limit 1000 --json databaseId -q '.[].databaseId' | wc -l)
    
    if [ "$count" -eq 0 ]; then
        echo -e "${YELLOW}削除対象がありません${NC}"
        exit 0
    fi
    
    echo "削除予定の実行数: $count"
    confirm_deletion "削除を実行します..."
    
    echo -e "${GREEN}削除中...${NC}"
    gh run list $workflow_filter --limit 1000 --json databaseId -q '.[].databaseId' | \
        xargs -I {} sh -c 'yes | gh run delete {}'
    
    echo -e "${GREEN}✓ 削除完了${NC}"
}

# メイン処理
delete_runs