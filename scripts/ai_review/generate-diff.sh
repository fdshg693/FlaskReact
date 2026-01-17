#!/usr/bin/env bash
# Generate Git Diff Script - Wrapper for Python Implementation
# 
# This is a backward compatibility wrapper that calls the Python version.
# For new usage, please use: python generate_diff.py
#
# Required environment variables:
#   - GITHUB_OUTPUT: Path to GitHub Actions output file
#
# Optional environment variables:
#   - PR_BASE_REF: Pull request base branch reference
#   - INPUT_TARGET: Target branch from workflow inputs
#
# Arguments:
#   None (uses environment variables)
#
# Outputs:
#   - tmp/diff.patch: Generated diff file
#   - Sets GITHUB_OUTPUT: has_changes=true/false

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "⚠️  Note: This is a backward compatibility wrapper."
echo "    For direct usage, please use: python generate_diff.py"
echo ""

# Determine the base branch for comparison
PR_BASE_REF="${PR_BASE_REF:-}"
INPUT_TARGET="${INPUT_TARGET:-}"

if [ -n "$PR_BASE_REF" ] && [ "$PR_BASE_REF" != 'null' ]; then
  BASE_BRANCH="$PR_BASE_REF"
elif [ -n "$INPUT_TARGET" ] && [ "$INPUT_TARGET" != 'null' ]; then
  BASE_BRANCH="$INPUT_TARGET"
else
  BASE_BRANCH='main'
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

# Call Python implementation
cd "$SCRIPT_DIR"
$PYTHON_CMD generate_diff.py -b "$BASE_BRANCH" -o tmp/diff.patch -v

# Check exit code
if [ $? -ne 0 ]; then
  echo "❌ Error: Python implementation failed"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "has_changes=false" >> "$GITHUB_OUTPUT"
  fi
  exit 1
fi

# Check if diff has content
if [ ! -s tmp/diff.patch ]; then
  echo "ℹ️ No changes detected"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "has_changes=false" >> "$GITHUB_OUTPUT"
  fi
  exit 0
fi

# Success
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "has_changes=true" >> "$GITHUB_OUTPUT"
fi
echo "✅ Diff generation completed successfully"
