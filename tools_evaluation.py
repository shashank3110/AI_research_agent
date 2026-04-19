from langchain_core.tools import tool
import os

@tool
def read_file(filename: str) -> str:
    """Reads the contents of a file (e.g. logs, csv, json) from the current directory."""
    run_dir = os.environ.get("CURRENT_RUN_DIR", ".")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(run_dir, safe_filename)
    
    if not os.path.exists(filepath):
        return f"Error: File {filepath} does not exist."
    try:
        with open(filepath, 'r') as f:
            # Safely truncate file outputs to prevent destroying the API token limit
            content = f.read()
            if len(content) > 3000:
                return content[:3000] + "\n...[TRUNCATED to save token limit]"
            return content
    except Exception as e:
        return f"Failed to read file: {str(e)}"
