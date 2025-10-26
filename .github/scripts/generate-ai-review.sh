#!/usr/bin/env bash
# AI Code Review Generator Script
# This script sends a diff to OpenAI API and generates a code review
# 
# Required environment variables:
#   - OPENAI_API_KEY: OpenAI API key
#   - AI_MODEL: Model to use (e.g., gpt-4)
#   - MAX_TOKENS: Maximum tokens for the response
#   - TEMPERATURE: Temperature for the model
#   - GITHUB_OUTPUT: Path to GitHub Actions output file
#
# Optional environment variables:
#   - REVIEW_PROMPT: Custom review prompt (if not set, uses default)
#
# Arguments:
#   $1: Path to diff file

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
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ Error: OPENAI_API_KEY environment variable is not set"
  exit 1
fi

if [ -z "${AI_MODEL:-}" ]; then
  echo "❌ Error: AI_MODEL environment variable is not set"
  exit 1
fi

if [ -z "${MAX_TOKENS:-}" ]; then
  echo "❌ Error: MAX_TOKENS environment variable is not set"
  exit 1
fi

if [ -z "${TEMPERATURE:-}" ]; then
  echo "❌ Error: TEMPERATURE environment variable is not set"
  exit 1
fi

if [ -z "${GITHUB_OUTPUT:-}" ]; then
  echo "❌ Error: GITHUB_OUTPUT environment variable is not set"
  exit 1
fi

# Validate diff file argument
if [ $# -lt 1 ]; then
  echo "❌ Error: Diff file path is required as first argument"
  exit 1
fi

DIFF_FILE="$1"

if [ ! -f "$DIFF_FILE" ]; then
  echo "❌ Error: Diff file not found: $DIFF_FILE"
  exit 1
fi

# Create prompt file
# mktemp creates a temporary file and returns its path (e.g., /tmp/tmp.abcd1234)
# This file will be automatically cleaned up when the script exits (see trap below)
TEMP_PROMPT=$(mktemp)
# trap: When the script exits (EXIT event), run 'rm -f "$TEMP_PROMPT"' to delete the temp file
# This ensures cleanup even if the script fails or is interrupted
trap 'rm -f "$TEMP_PROMPT"' EXIT

# Use provided prompt or default
# ${REVIEW_PROMPT:-} checks if REVIEW_PROMPT is set:
# - If REVIEW_PROMPT is set and non-empty: use its value
# - If REVIEW_PROMPT is not set or empty: use "" (empty string)
# The second [ "$REVIEW_PROMPT" != '' ] checks if the result is non-empty
# This allows using a custom prompt from environment variables, or falling back to the default
if [ -n "${REVIEW_PROMPT:-}" ] && [ "$REVIEW_PROMPT" != '' ]; then
  # If custom prompt is provided, use it directly
  echo "$REVIEW_PROMPT" > "$TEMP_PROMPT"
else
  # Otherwise, use the default review prompt template
  cat > "$TEMP_PROMPT" << 'EOF'
You are an experienced software engineer. Please review the following code diff in detail and analyze it from the following perspectives in English:

1. Code Quality: Readability, maintainability, performance
2. Security: Potential vulnerabilities and security risks
3. Best Practices: Language and framework recommendations
4. Bug Potential: Logic errors and exception handling issues
5. Improvement Suggestions: Specific improvement proposals and refactoring suggestions

Output Format:
- Point out issues specifically and include relevant line numbers
- Provide implementable concrete examples for improvement suggestions
- Clearly indicate importance level (High, Medium, Low)

Code Diff:
EOF
fi

# Append the actual diff content to the prompt
echo "" >> "$TEMP_PROMPT"
cat "$DIFF_FILE" >> "$TEMP_PROMPT"

# Create API request payload
# Creating another temporary file for the JSON payload to send to OpenAI API
TEMP_PAYLOAD=$(mktemp)
# trap: Update to handle multiple temp files
# Multiple trap commands add to the cleanup list (don't overwrite the previous one)
trap 'rm -f "$TEMP_PROMPT" "$TEMP_PAYLOAD"' EXIT

jq -n \
  --arg model "$AI_MODEL" \
  --rawfile prompt "$TEMP_PROMPT" \
  --argjson max_tokens "$MAX_TOKENS" \
  --argjson temperature "$TEMPERATURE" \
  '{
    model: $model,
    messages: [
      {
        role: "system",
        content: "You are a helpful and constructive code reviewer. Please provide detailed and practical feedback."
      },
      {
        role: "user",
        content: $prompt
      }
    ],
    max_tokens: $max_tokens,
    temperature: $temperature
  }' > "$TEMP_PAYLOAD"

echo "🔄 Sending request to OpenAI API (model: $AI_MODEL)..."

# Send API request
# Creating a third temporary file to store the API response
API_RESPONSE=$(mktemp)
# trap: Update to cleanup all three temporary files
trap 'rm -f "$TEMP_PROMPT" "$TEMP_PAYLOAD" "$API_RESPONSE"' EXIT

HTTP_CODE=$(curl -w "%{http_code}" -s \
  -X POST \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d @"$TEMP_PAYLOAD" \
  -o "$API_RESPONSE" \
  https://api.openai.com/v1/chat/completions
)

# Check HTTP status
if [ "$HTTP_CODE" -ne 200 ]; then
  echo "❌ Error: OpenAI API request failed with HTTP $HTTP_CODE"
  ERROR_TYPE=$(jq -r '.error.type // "unknown"' "$API_RESPONSE" 2>/dev/null || echo "unknown")
  ERROR_MESSAGE=$(jq -r '.error.message // "API call failed"' "$API_RESPONSE" 2>/dev/null || echo "API call failed")
  echo "Error type: $ERROR_TYPE"
  echo "Error message: $ERROR_MESSAGE"
  exit 1
fi

# Extract review content
REVIEW_CONTENT=$(jq -r '.choices[0].message.content // "No review content received"' "$API_RESPONSE" 2>/dev/null)

if [ "$REVIEW_CONTENT" = "null" ] || [ "$REVIEW_CONTENT" = "No review content received" ] || [ -z "$REVIEW_CONTENT" ]; then
  echo "❌ Error: No valid review content received from OpenAI"
  API_ERROR=$(jq -r '.error.message // "Unknown API error"' "$API_RESPONSE" 2>/dev/null || echo "Response parsing failed")
  echo "API indicated: $API_ERROR"
  exit 1
fi

# Write to GitHub Actions output
{
  echo "review<<REVIEW_EOF"
  echo "$REVIEW_CONTENT"
  echo "REVIEW_EOF"
} >> "$GITHUB_OUTPUT"

# Also save to file for artifact upload
echo "$REVIEW_CONTENT" > tmp/ai_review_output.md

echo "✅ AI review generated successfully"
