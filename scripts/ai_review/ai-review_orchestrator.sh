#!/usr/bin/env bash
# Generate AI Code Review Script - Wrapper for Python Implementation
#
# This is a backward compatibility wrapper that calls the Python version.
# For new usage, please use: python ai_review_orchestrator.py
#
# Usage:
#   ./ai-review_orchestrator.sh [base_branch]
#
# Arguments:
#   base_branch: Optional. The base branch to compare against (default: main)
#
# Required files:
#   - .env: Environment variables file (OPENAI_API_KEY required)
#
# Outputs:
#   - tmp/diff.patch: Generated diff file
#   - tmp/ai_review_output.md: AI-generated review

set -euo pipefail

# Get the script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

echo "⚠️  Note: This is a backward compatibility wrapper."
echo "    For direct usage, please use: python ai_review_orchestrator.py"
echo ""

# Determine base branch
BASE_BRANCH="${1:-main}"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
  echo "❌ Error: Python is not installed"
  echo "Please install Python 3.8+ to use this tool"
  exit 1
fi

# Detect Python command
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
  PYTHON_CMD="python3"
fi

# Load environment variables from .env file if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
  echo "📁 Loading environment variables from .env..."
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
elif [ -f "$SCRIPT_DIR/.env" ]; then
  echo "📁 Loading environment variables from .env..."
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Call Python implementation
$PYTHON_CMD ai_review_orchestrator.py "$BASE_BRANCH" -v

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "❌ Error: Python implementation failed"
  exit $EXIT_CODE
fi

echo "✅ AI Code Review completed successfully"