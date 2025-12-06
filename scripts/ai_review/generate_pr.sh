#!/usr/bin/env bash
# Generate AI Code Review Script
# This script generates a diff and creates an AI code review using OpenAI API
#
# Usage:
#   ./scripts/generate_pr.sh [base_branch]
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

# Change to project root
cd "$PROJECT_ROOT"

echo "🚀 Starting AI Code Review Process"
echo "================================================"

# Load environment variables from .env file
if [ ! -f .env ]; then
  echo "❌ Error: .env file not found in project root"
  echo "Please create .env file based on sample.env"
  exit 1
fi

echo "📁 Loading environment variables from .env..."
# Export variables from .env file
# Using set -a to automatically export all variables
set -a
source .env
set +a

# Validate required environment variables
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ Error: OPENAI_API_KEY is not set in .env file"
  exit 1
fi

# Set default values for AI model configuration
export AI_MODEL="${AI_MODEL:-gpt-4.1}"
export MAX_TOKENS="${MAX_TOKENS:-10000}"
export TEMPERATURE="${TEMPERATURE:-0.1}"

echo "✅ Environment variables loaded"
echo "   - Model: $AI_MODEL"
echo "   - Max Tokens: $MAX_TOKENS"
echo "   - Temperature: $TEMPERATURE"
echo ""

# Determine base branch
BASE_BRANCH="${1:-main}"
echo "📊 Base branch for comparison: $BASE_BRANCH"

# Create temporary GITHUB_OUTPUT file for compatibility with existing scripts
TEMP_OUTPUT=$(mktemp)
trap 'rm -f "$TEMP_OUTPUT"' EXIT
export GITHUB_OUTPUT="$TEMP_OUTPUT"

# Step 1: Generate diff using generate-diff.sh
echo ""
echo "================================================"
echo "📝 Step 1: Generating diff..."
echo "================================================"

# Set variables needed by generate-diff.sh
export INPUT_TARGET="$BASE_BRANCH"

# Run the generate-diff script
if [ -f ".github/scripts/generate-diff.sh" ]; then
  bash .github/scripts/generate-diff.sh
else
  echo "❌ Error: .github/scripts/generate-diff.sh not found"
  exit 1
fi

# Check if diff was generated
if [ ! -f tmp/diff.patch ]; then
  echo "❌ Error: tmp/diff.patch was not created"
  exit 1
fi

# Check if there are changes
HAS_CHANGES=$(grep "has_changes=" "$GITHUB_OUTPUT" | cut -d'=' -f2 || echo "false")

if [ "$HAS_CHANGES" = "false" ]; then
  echo "ℹ️ No changes detected. Skipping AI review."
  exit 0
fi

echo "✅ Diff generated successfully"
echo ""

# Step 2: Generate AI review using generate-ai-review.sh
echo "================================================"
echo "🤖 Step 2: Generating AI review..."
echo "================================================"

# Run the AI review script
if [ -f ".github/scripts/generate-ai-review.sh" ]; then
  bash .github/scripts/generate-ai-review.sh tmp/diff.patch
else
  echo "❌ Error: .github/scripts/generate-ai-review.sh not found"
  exit 1
fi

# Check if review was generated
if [ ! -f tmp/ai_review_output.md ]; then
  echo "❌ Error: tmp/ai_review_output.md was not created"
  exit 1
fi

echo ""
echo "================================================"
echo "✅ AI Code Review Completed!"
echo "================================================"
echo ""
echo "📄 Generated files:"
echo "   - tmp/diff.patch: Git diff between $BASE_BRANCH and current branch"
echo "   - tmp/ai_review_output.md: AI-generated code review"
echo ""
echo "📖 View the review:"
echo "   cat tmp/ai_review_output.md"
echo ""