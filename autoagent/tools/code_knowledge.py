import os
import subprocess
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from autoagent.registry import register_tool

logger = logging.getLogger(__name__)

@register_tool(name="gen_code_tree_structure")
def gen_code_tree_structure(
    root_path: str, 
    exclude_dirs: Optional[List[str]] = None,
    exclude_files: Optional[List[str]] = None,
    max_depth: int = 5,
    show_files: bool = True,
    show_content: bool = False,
    file_types: Optional[List[str]] = None
) -> str:
    """
    Generate a tree structure of code files in a directory.
    
    Args:
        root_path: The root directory to start the tree from.
        exclude_dirs: List of directory names to exclude.
        exclude_files: List of file names to exclude.
        max_depth: Maximum depth to traverse.
        show_files: Whether to include individual files in the output.
        show_content: Whether to include file content (limited preview).
        file_types: List of file extensions to include (e.g., ['.py', '.js']).
        
    Returns:
        A string representation of the directory tree structure.
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode']
    if exclude_files is None:
        exclude_files = []
    if file_types is None:
        file_types = []
        
    root_path = os.path.abspath(root_path)
    if not os.path.exists(root_path):
        return f"Error: Path '{root_path}' does not exist."
    
    result = []
    result.append(f"Directory Tree for: {root_path}")
    result.append("=" * 40)
    
    def should_process_file(file_name, extensions):
        if not extensions:
            return True
        return any(file_name.endswith(ext) for ext in extensions)
    
    def process_directory(current_path, prefix="", depth=0):
        if depth > max_depth:
            result.append(f"{prefix}│")
            result.append(f"{prefix}└── [Max depth reached]")
            return
            
        items = sorted(os.listdir(current_path))
        dirs = [item for item in items if os.path.isdir(os.path.join(current_path, item)) and item not in exclude_dirs]
        files = [item for item in items if os.path.isfile(os.path.join(current_path, item)) and item not in exclude_files]
        
        for i, dirname in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            dir_path = os.path.join(current_path, dirname)
            
            if is_last_dir:
                result.append(f"{prefix}└── {dirname}/")
                process_directory(dir_path, f"{prefix}    ", depth + 1)
            else:
                result.append(f"{prefix}├── {dirname}/")
                process_directory(dir_path, f"{prefix}│   ", depth + 1)
                
        if show_files:
            for i, filename in enumerate(files):
                if not should_process_file(filename, file_types):
                    continue
                    
                is_last = (i == len(files) - 1)
                file_path = os.path.join(current_path, filename)
                
                if is_last:
                    result.append(f"{prefix}└── {filename}")
                    file_prefix = f"{prefix}    "
                else:
                    result.append(f"{prefix}├── {filename}")
                    file_prefix = f"{prefix}│   "
                
                if show_content:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(500)  # Read only first 500 chars
                        
                        if content:
                            content_lines = content.split('\n')[:5]  # Only first 5 lines
                            for j, line in enumerate(content_lines):
                                if j < len(content_lines) - 1:
                                    result.append(f"{file_prefix}│ {line}")
                                else:
                                    if len(content) > 500 or len(content.split('\n')) > 5:
                                        result.append(f"{file_prefix}│ {line}")
                                        result.append(f"{file_prefix}└── [File content truncated]")
                                    else:
                                        result.append(f"{file_prefix}└── {line}")
                    except Exception as e:
                        result.append(f"{file_prefix}└── [Error reading file: {str(e)}]")
    
    try:
        process_directory(root_path)
    except Exception as e:
        logger.error(f"Error generating code tree: {str(e)}")
        result.append(f"Error occurred: {str(e)}")
    
    return "\n".join(result)