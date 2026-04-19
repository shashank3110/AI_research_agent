"""
Implementation Tools for the AI Research Agent.

Provides python script execution, file writing tools, and a tool to ask
the human operator for help when the autonomous implementation is stuck.
"""
from langchain_core.tools import tool
import subprocess
import os

@tool
def ask_human(question: str) -> str:
    """Useful when you need clarification from the human user on how to access a dataset or library constraint. Pauses execution and awaits user input."""
    print(f"\n\n====================================")
    print(f"[IMPLEMENTATION AGENT NEEDS HELP]: {question}")
    answer = input("Provide your guidance/answer: ")
    print(f"====================================\n")
    return answer

@tool
def write_python_script(filename: str, code_content: str) -> str:
    """Writes Python code to a specific file in the current directory."""
    run_dir = os.environ.get("CURRENT_RUN_DIR", ".")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(run_dir, safe_filename)
    try:
        with open(filepath, 'w') as f:
            f.write(code_content)
        return f"Successfully wrote code to {filepath}"
    except Exception as e:
        return f"Failed to write file: {str(e)}"

@tool
def execute_python_script(filename: str, timeout_seconds: int = 120) -> str:
    """Executes a given python file returning stdout/stderr. Used for training or evaluating."""
    run_dir = os.environ.get("CURRENT_RUN_DIR", ".")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(run_dir, safe_filename)
    
    # Ensure file exists
    if not os.path.exists(filepath):
        return f"Error: File {filepath} does not exist."
    try:
        result = subprocess.run(
            ['python3', safe_filename], 
            cwd=run_dir,
            capture_output=True, 
            text=True, 
            timeout=timeout_seconds
        )
        return f"Return Code: {result.returncode}\nStdout: {result.stdout[:2000]}\nStderr: {result.stderr[:2000]}"
    except subprocess.TimeoutExpired:
        return "Execution failed: Script timed out."
    except Exception as e:
        return f"Execution failed: {str(e)}"
