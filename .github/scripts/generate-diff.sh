#!/usr/bin/env bash
# Generate Git Diff Script
# This script generates a diff between the current branch and a base branch
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
# NOTE: Bash/Shell scripting terminology:
# - mktemp: Creates a temporary file with a unique name. Returns the file path.
#           Used here to create temporary files for the prompt and API payload.
#           Example: TEMP_PROMPT=$(mktemp) creates /tmp/tmp.XXXXXXXXXX
# 
# - trap: Registers a command to run when the script exits (normally or on error).
#         Syntax: trap 'command' EXIT
#         Used here to ensure temporary files are deleted after use.
#         Example: trap 'rm -f "$TEMP_PROMPT"' EXIT removes $TEMP_PROMPT on exit.
#
# - ${VARIABLE:-default}: Parameter expansion with default value.
#         Returns VARIABLE if set and non-empty, otherwise returns 'default'.
#         Syntax: ${VARIABLE:-default_value}
#         Example: ${OPENAI_API_KEY:-} returns "" if OPENAI_API_KEY is not set.
#         Used here to safely check if environment variables are set.

# Validate required environment variables
if [ -z "${GITHUB_OUTPUT:-}" ]; then
  echo "❌ Error: GITHUB_OUTPUT environment variable is not set"
  exit 1
fi

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

echo "Comparing against base branch: $BASE_BRANCH"

# Fetch the base branch
git fetch origin "$BASE_BRANCH" || {
  echo "❌ Error: Failed to fetch base branch '$BASE_BRANCH'"
  echo "Available branches:"
  git branch -r
  exit 1
}

# Generate diff
git diff "origin/$BASE_BRANCH...HEAD" \
  --unified=3 \
  --no-color \
  --ignore-space-change \
  > tmp/diff.patch

# Check if diff has content
if [ ! -s tmp/diff.patch ]; then
  echo "ℹ️ No changes detected between current branch and $BASE_BRANCH"
  echo "has_changes=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "has_changes=true" >> "$GITHUB_OUTPUT"
echo "✅ Generated diff file ($(wc -l < tmp/diff.patch) lines)"
