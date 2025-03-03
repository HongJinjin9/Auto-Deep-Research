import click
import importlib
from autoagent import MetaChain
from autoagent.util import debug_print
import asyncio
from autoagent.logger import LoggerManager, MetaChainLogger 
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.panel import Panel
import re
from rich.progress import Progress, SpinnerColumn, TextColumn
from autoagent.util import ask_text, single_select_menu, print_markdown, debug_print, UserCompleter
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from autoagent.agents import get_system_triage_agent
from loop_utils.font_page import MC_LOGO, version_table, NOTES, GOODBYE_LOGO
from rich.live import Live
from autoagent.environment.docker_env import DockerEnv, DockerConfig, check_container_ports, is_docker_available, LocalDockerEmulator
from autoagent.environment.browser_env import BrowserEnv
from autoagent.environment.markdown_browser import RequestsMarkdownBrowser
import os
import os.path as osp
import socket
from constant import DOCKER_WORKPLACE_NAME, COMPLETION_MODEL

# Add Docker availability check function
logger = logging.getLogger("cli")
docker_available = is_docker_available()
if not docker_available:
    logger.warning("Docker is not available, falling back to local environment")

def check_port_available(port):
    """check if the port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # set the port reuse option
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # try to bind the port
            s.bind(('0.0.0.0', port))
            # immediately close the connection
            s.close()
            return True  # the port is available
        except socket.error:
            return False  # the port is not available

@click.group()
def cli():
    """The command line interface for autoagent"""
    pass

def clear_screen():
    console = Console()
    console.print("[bold green]Coming soon...[/bold green]")
    print('\033[u\033[J\033[?25h', end='')  # Restore cursor and clear everything after it, show cursor

def get_config(container_name, port):
    from autoagent.environment.docker_env import DockerConfig, check_container_ports, is_docker_available
    from autoagent.environment.local_env import LocalDockerEmulator
    import logging

    docker_available = is_docker_available()
    logger = logging.getLogger("cli")
    
    try:
        if docker_available:
            port_info = check_container_ports(container_name)
            if port_info:
                port = port_info[0]
        else:
            logger.warning("Docker is not available, falling back to local environment")
            port = port
            
        local_root = os.path.join(os.getcwd(), f"workspace_meta_showcase", f"showcase_{container_name}")
        os.makedirs(local_root, exist_ok=True)

        docker_config = DockerConfig(
            workplace_name=DOCKER_WORKPLACE_NAME,
            container_name=container_name,
            communication_port=port,
            conda_path='/root/miniconda3',
            local_root=local_root,
        )

        if not docker_available:
            docker_config.use_emulator = True
            
        return docker_config
            
    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}")
        if not docker_available:
            logger.error("Docker is required but not available. Please install Docker or use a different port.")
        raise

def create_environment(docker_config: DockerConfig):
    """
    1. create the code environment
    2. create the web environment
    3. create the file environment
    """
    try:
        docker_available = is_docker_available()
        
        if docker_available:
            code_env = DockerEnv(docker_config)
            code_env.init_container()
        else:
            logger.warning("Docker is not available, creating local environment")
            code_env = LocalDockerEmulator(docker_config)
            code_env.start_container()
            
        web_env = BrowserEnv(browsergym_eval_env=None, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name)
        file_env = RequestsMarkdownBrowser(viewport_size=1024 * 5, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name, downloads_folder=os.path.join(docker_config.local_root, docker_config.workplace_name, "downloads"))
        
        return code_env, web_env, file_env
        
    except Exception as e:
        logger.error(f"Failed to create environment: {str(e)}")
        if not docker_available:
            logger.error("Docker is required but not available. Please install Docker or use a different port.")
        raise

def deep_research(container_name: str, port: int):
    try:
        if not is_docker_available():
            logger.warning("Docker is not available, using local environment")
            
        print("port: ", port)
        docker_config = get_config(container_name, port)
        code_env, web_env, file_env = create_environment(docker_config)
        
        logger.info("Starting deep research with container name: %s and port: %d", container_name, port)
        
        context_variables = {"working_dir": docker_config.workplace_name, "code_env": code_env, "web_env": web_env, "file_env": file_env}

        progress.update(task, description="[cyan]Setting up autoagent...[/cyan]\n")

        clear_screen()

        style = Style.from_dict({
            'bottom-toolbar': 'bg:#333333 #ffffff',
        })

        # 创建会话
        session = PromptSession(
            completer=UserCompleter(context_variables.keys()),
            complete_while_typing=True,
            style=style
        )
        client = MetaChain(log_path=LoggerManager.get_logger())
        while True: 
            # query = ask_text("Tell me what you want to do:")
            query = session.prompt(
                'Tell me what you want to do (type "exit" to quit): ',
                bottom_toolbar=HTML('<b>Prompt:</b> Enter <b>@</b> to mention Agents')
            )
            if query.strip().lower() == 'exit':
                # logger.info('User mode completed.  See you next time! :waving_hand:', color='green', title='EXIT')
                logo_text = "See you next time! :waving_hand:"
                console.print(Panel(logo_text, style="bold salmon1", expand=True))
                break
            words = query.split()
            console.print(f"[bold green]Your request: {query}[/bold green]", end=" ")
            for word in words:
                if word.startswith('@') and word[1:] in context_variables.keys():
                    agent = context_variables[word.replace('@', '')]
                else:
                    pass
            print()
            
            if hasattr(agent, "name"): 
                agent_name = agent.name
                console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] will help you, be patient...[/bold green]")
                messages.append({"role": "user", "content": query})
                response = client.run(agent, messages, context_variables, debug=False)
                messages.extend(response.messages)
                model_answer_raw = response.messages[-1]['content']
                
                # attempt to parse model_answer
                if model_answer_raw.startswith('Case resolved'):
                    model_answer = re.findall(r'<solution>(.*?)</solution>', model_answer_raw, re.DOTALL)
                    if len(model_answer) == 0:
                        model_answer = model_answer_raw
                    else:
                        model_answer = model_answer[0]
                else: 
                    model_answer = model_answer_raw
                console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] has finished with the response:\n[/bold green] [bold blue]{model_answer}[/bold blue]")
                agent = response.agent
            elif agent == "select": 
                code_env: DockerEnv = context_variables["code_env"]
                local_workplace = code_env.local_workplace
                files_dir = os.path.join(local_workplace, "files")
                os.makedirs(files_dir, exist_ok=True)
                select_and_copy_files(files_dir, console)
            else: 
                console.print(f"[bold red]Unknown agent: {agent}[/bold red]")
    except Exception as e:
        logger.error(f"Deep research failed: {str(e)}")
        if not docker_available:
            logger.error("Docker is required but not available. Please install Docker or use a different port.")
        raise