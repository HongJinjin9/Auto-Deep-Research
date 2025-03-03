import os
import importlib
from autoagent.registry import registry

# Explicitly import the system triage agent
from autoagent.agents.system_agent.system_triage_agent import get_system_triage_agent
# Import other system agents
from autoagent.agents.system_agent.filesurfer_agent import get_filesurfer_agent
from autoagent.agents.system_agent.programming_agent import get_coding_agent
from autoagent.agents.system_agent.websurfer_agent import get_websurfer_agent

def import_agents_recursively(base_dir: str, base_package: str):
    """Recursively import all agents in .py files
    
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

# get the current directory and import all agents
current_dir = os.path.dirname(__file__)
import_agents_recursively(current_dir, 'autoagent.agents')

# export all agent creation functions
globals().update(registry.agents)
globals().update(registry.plugin_agents)

# Add system agents explicitly to __all__
__all__ = list(registry.agents.keys()) + [
    'get_system_triage_agent',
    'get_filesurfer_agent',
    'get_coding_agent',
    'get_websurfer_agent'
]