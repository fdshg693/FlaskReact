#!/bin/bash

# -----------------------------------------------------------------------------
# Script Name: scripts.sh
# Description: Run AI reviews in parallel based on a YAML configuration file.
# Usage: ./scripts.sh <settings_yaml_file>
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 <settings_yaml_file>"
    echo ""
    echo "Arguments:"
    echo "  <settings_yaml_file>  Path to the YAML settings file (e.g., scripts/github_copilot/settings/review.yaml)"
    echo ""
    echo "Description:"
    echo "  This script reads the specified YAML file to get a list of root directories,"
    echo "  an agent name, and a prompt template. It then runs 'copilot' reviews"
    echo "  for each directory in parallel."
    exit 1
}

# Check if the required tools are installed
check_dependencies() {
    local missing_tools=0
    for tool in yq envsubst copilot; do
        if ! command -v "$tool" &> /dev/null; then
            echo "Error: Required tool '$tool' is not installed or not in PATH."
            missing_tools=1
        fi
    done
    
    if [ $missing_tools -eq 1 ]; then
        echo "Please install the missing tools and try again."
        exit 1
    fi
}

# Main execution
main() {
    # Check for arguments
    if [ $# -eq 0 ]; then
        usage
    fi

    local settings_file="$1"

    # Check if settings file exists
    if [ ! -f "$settings_file" ]; then
        echo "Error: Settings file '$settings_file' not found."
        exit 1
    fi

    check_dependencies

    echo "Loading settings from $settings_file..."

    # Extract configuration using yq (using -r for raw output)
    if ! agent=$(yq -r '.agent' "$settings_file"); then
        echo "Error: Failed to parse 'agent' from YAML."
        exit 1
    fi
    
    if ! raw_prompt=$(yq -r '.prompt' "$settings_file"); then
        echo "Error: Failed to parse 'prompt' from YAML."
        exit 1
    fi

    # Read root_dirs into an array
    root_dirs=()
    while IFS= read -r line; do
        if [ -n "$line" ] && [ "$line" != "null" ]; then
            root_dirs+=("$line")
        fi
    done < <(yq -r '.root_dirs[]' "$settings_file")

    if [ ${#root_dirs[@]} -eq 0 ]; then
        echo "Error: No 'root_dirs' found in settings file."
        exit 1
    fi

    echo "Agent: $agent"
    echo "Target Directories: ${#root_dirs[@]}"
    echo "---------------------------------------------------"

    # Create a log directory for this run
    # Use absolute path to avoid issues when changing directories
    current_dir=$(pwd)
    log_dir="${current_dir}/logs/copilot_review_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$log_dir"
    echo "Logs will be saved to: $log_dir"

    # Trap SIGINT (Ctrl+C) to kill background processes
    trap 'echo -e "\nAborting..."; kill 0' SIGINT

    pids=()
    
    for root_dir in "${root_dirs[@]}"; do
        (
            # Run in a subshell
            # echo "Starting review for: $root_dir" # Removed to reduce noise

            if [ ! -d "$root_dir" ]; then
                echo "Warning: Directory '$root_dir' does not exist. Skipping."
                exit 0
            fi

            # Prepare the prompt with variable substitution
            # Export root_dir so envsubst can use it
            export root_dir="$root_dir"
            # Substitute only ${root_dir}
            prompt=$(echo "$raw_prompt" | envsubst '${root_dir}')

            # Define log file path (replace slashes with underscores for filename)
            safe_dirname=$(echo "$root_dir" | tr '/' '_')
            log_file="${log_dir}/${safe_dirname}.log"
            
            # Create the log file explicitly
            touch "$log_file"

            # Log execution details
            {
                echo "=== Execution Details ==="
                echo "Timestamp: $(date)"
                echo "Directory: $root_dir"
                echo "Agent: $agent"
                echo "--- Prompt ---"
                echo "$prompt"
                echo "--- Command ---"
                echo "copilot --agent=\"$agent\" --prompt=\"$prompt\" --allow-all-tools"
                echo "========================="
                echo ""
            } > "$log_file"

            # Execute copilot command
            # Redirect stdin from /dev/null to prevent interactive prompts
            # Redirect stdout and stderr to the log file
            if copilot --agent="$agent" --prompt="$prompt" --allow-all-tools < /dev/null >> "$log_file" 2>&1; then
                echo "✅ [$root_dir] Review completed."
            else
                echo "❌ [$root_dir] Review failed. Check log: $log_file"
                exit 1
            fi
        ) &
        
        # Store the PID of the background process
        pids+=($!)
        echo "🚀 [$root_dir] Started review (PID: $!)"
    done

    # Wait for all background processes to finish
    echo "Reviews are running in parallel. Waiting for completion..."
    
    failures=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done

    echo "---------------------------------------------------"
    if [ $failures -eq 0 ]; then
        echo "🎉 All reviews completed successfully."
    else
        echo "⚠️  $failures review(s) encountered errors."
        exit 1
    fi
}

main "$@"
