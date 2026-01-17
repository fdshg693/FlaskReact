#!/usr/bin/env bash
# AI Code Review Generator Script - Wrapper for Python Implementation
# 
# This is a backward compatibility wrapper that calls the Python version.
# For new usage, please use: python generate_ai_review.py
#
# Required environment variables:
#   - OPENAI_API_KEY: OpenAI API key
#
# Optional environment variables:
#   - AI_MODEL: Model to use (default: gpt-4o)
#   - MAX_TOKENS: Maximum tokens for the response (default: 10000)
#   - TEMPERATURE: Temperature for the model (default: 0.1)
#   - REVIEW_PROMPT: Custom review prompt (if not set, uses default)
#
# Arguments:
#   $1: Path to diff file (default: tmp/diff.patch)

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "⚠️  Note: This is a backward compatibility wrapper."
echo "    For direct usage, please use: python generate_ai_review.py"
echo ""

# Get diff file path from argument or use default
DIFF_FILE="${1:-tmp/diff.patch}"

# Validate required environment variables
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ Error: OPENAI_API_KEY environment variable is not set"
  echo "Please set it in .env file or environment"
  exit 1
fi

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

# Build Python command with optional arguments
PYTHON_ARGS=("$DIFF_FILE")

# Add optional model argument
if [ -n "${AI_MODEL:-}" ]; then
  PYTHON_ARGS+=("--model" "$AI_MODEL")
fi

# Add custom prompt if provided
if [ -n "${REVIEW_PROMPT:-}" ]; then
  # Create temporary prompt file
  TEMP_PROMPT=$(mktemp)
  trap 'rm -f "$TEMP_PROMPT"' EXIT
  echo "$REVIEW_PROMPT" > "$TEMP_PROMPT"
  PYTHON_ARGS+=("--prompt-file" "$TEMP_PROMPT")
fi

# Add other optional arguments
if [ -n "${MAX_TOKENS:-}" ]; then
  export MAX_TOKENS
fi

if [ -n "${TEMPERATURE:-}" ]; then
  export TEMPERATURE
fi

# Call Python implementation
cd "$SCRIPT_DIR"
$PYTHON_CMD generate_ai_review.py "${PYTHON_ARGS[@]}" -v

# Check exit code
if [ $? -ne 0 ]; then
  echo "❌ Error: Python implementation failed"
  exit 1
fi

# Write to GitHub Actions output if GITHUB_OUTPUT is set
if [ -n "${GITHUB_OUTPUT:-}" ] && [ -f tmp/ai_review_output.md ]; then
  {
    echo "review<<REVIEW_EOF"
    cat tmp/ai_review_output.md
    echo "REVIEW_EOF"
  } >> "$GITHUB_OUTPUT"
fi

echo "✅ AI review generation completed successfully"
