from autoagent.registry import registry, register_tool, register_plugin_tool
import os
import importlib
from typing import Union, Optional
from autoagent.types import Result
from autoagent.environment.markdown_browser import RequestsMarkdownBrowser
from autoagent.environment import LocalEnv

# Explicit imports for commonly used tools
from .file_surfer_tool import (
    open_local_file,
    page_up_markdown,
    page_down_markdown,
    find_on_page_ctrl_f,
    find_next,
    visual_question_answering
)
from .terminal_tools import (
    execute_command,
    list_files,
    read_file,
    create_file,
    write_file,
    create_directory
)
from .web_tools import (
    web_search,
    download_url,
    click
)
from .rag_tools import (
    rag
)
from .rag_code import (
    code_rag
)
from .tool_utils import (
    get_current_time
)
from .code_knowledge import (
    gen_code_tree_structure
)

def import_tools_recursively(base_dir: str, base_package: str):
    """Recursively import all tools in .py files
    
    Args:
        base_dir: the root directory to start searching
        base_package: the base name of the Python package
    """
    for root, dirs, files in os.walk(base_dir):
        # get the relative path to the base directory
        rel_path = os.path.relpath(root, base_dir)
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # build the module path
                if rel_path == '.':
                    # in the root directory
                    module_path = f"{base_package}.{file[:-3]}"
                else:
                    # in the subdirectory
                    package_path = rel_path.replace(os.path.sep, '.')
                    module_path = f"{base_package}.{package_path}.{file[:-3]}"
                
                try:
                    importlib.import_module(module_path)
                except Exception as e:
                    print(f"Warning: Failed to import {module_path}: {e}")

# get the current directory and import all tools
current_dir = os.path.dirname(__file__)
import_tools_recursively(current_dir, 'autoagent.tools')

# export all tool creation functions
globals().update(registry.tools)
globals().update(registry.plugin_tools)

__all__ = list(registry.tools.keys()) + list(registry.plugin_tools.keys())