"""
Scientific Writing Tools for the AI Research Agent.

Provides tools to save the final documents (Markdown, LaTeX) and to 
compile LaTeX source code into PDF format.
"""
from langchain_core.tools import tool
import os
import subprocess

@tool
def write_document(filename: str, content: str) -> str:
    """Writes the markdown or latex content to a file. Useful for saving the final scientific paper."""
    run_dir = os.environ.get("CURRENT_RUN_DIR", ".")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(run_dir, safe_filename)
    
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Document successfully saved to {filepath}"
    except Exception as e:
        return f"Failed to write document: {str(e)}"

@tool
def compile_latex(filename: str) -> str:
    """Compiles a .tex file into a .pdf using pdflatex."""
    run_dir = os.environ.get("CURRENT_RUN_DIR", ".")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(run_dir, safe_filename)
    
    if not safe_filename.endswith('.tex'):
        return "Error: File must be a .tex file."
    try:
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', safe_filename], cwd=run_dir, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully compiled {filepath} to PDF."
        else:
            return f"LaTeX compilation issue (PDF might still be generated). Return code: {result.returncode}\nStdout: {result.stdout[:1000]}"
    except FileNotFoundError:
        return "Error: pdflatex command not found. Ensure TeX Live or MiKTeX is installed on the system."
    except Exception as e:
        return f"Failed to compile LaTeX: {str(e)}"
