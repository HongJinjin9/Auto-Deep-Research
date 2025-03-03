import os
import os.path as osp
import subprocess
import time
import socket
import json
from pathlib import Path
import shutil
import logging
from typing import Optional, Union, Dict, Tuple
from functools import update_wrapper
from inspect import signature
import requests
from dataclasses import dataclass, field
from constant import BASE_IMAGES, GITHUB_AI_TOKEN  # Ensure this import is correct

@dataclass
class DockerConfig:
    container_name: str
    workplace_name: str
    communication_port: int = 12345
    conda_path: str = "/root/miniconda3"
    test_pull_name: str = field(default='main')
    task_name: Optional[str] = field(default=None)
    git_clone: bool = field(default=False)
    local_root: str = field(default_factory=lambda: os.getcwd())

# Add Docker availability check function
def is_docker_available() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Define a logger for docker_env
logger = logging.getLogger("docker_env")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if not logger.hasHandlers():
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# LocalDockerEmulator class for local fallback
class LocalDockerEmulator:
    def __init__(self, config: DockerConfig):
        self.config = config
        self.status = "running"
        self.ports = {config.communication_port: config.communication_port}

    def check_ports(self) -> Optional[Tuple[int, int]]:
        return (self.config.communication_port, self.config.communication_port)

    def is_container_running(self) -> bool:
        return self.status == "running"

    def get_container_ports(self) -> Optional[Tuple[int, int]]:
        return (self.config.communication_port, self.config.communication_port)

    def start_container(self) -> None:
        self.status = "running"
        logger.info("Started container using LocalDockerEmulator.")

    def stop_container(self) -> None:
        self.status = "stopped"
        logger.info("Stopped container using LocalDockerEmulator.")

    def run_command(self, command, stream_callback=None):
        logger.info(f"Running command '{command}' using LocalDockerEmulator.")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = ''
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            output += line
        error = process.stderr.read()
        if error:
            logger.error(f"Command '{command}' failed: {error}")
        return {
            "status": process.poll(),
            "result": output
        }

class DockerEnv:
    def __init__(self, config: Union[DockerConfig, Dict]):
        if isinstance(config, Dict):
            config = DockerConfig(**config)
        self.workplace_name = config.workplace_name
        self.local_workplace = osp.join(config.local_root, config.workplace_name)
        self.docker_workplace = f"/{config.workplace_name}"
        self.container_name = config.container_name
        self.test_pull_name = config.test_pull_name
        self.task_name = config.task_name
        self.git_clone = config.git_clone
        self.setup_package = config.setup_package
        self.communication_port = config.communication_port
        self.conda_path = config.conda_path
        self.use_emulator = not is_docker_available()
        if self.use_emulator:
            logger.warning("Docker is not available, using LocalDockerEmulator.")
            self.emulator = LocalDockerEmulator(config)

    def init_container(self):
        try:
            if self.use_emulator:
                self.emulator.start_container()
                logger.info("Container started using LocalDockerEmulator.")
                return

            container_check_command = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            existing_container = subprocess.run(container_check_command, capture_output=True, text=True)
            os.makedirs(self.local_workplace, exist_ok=True)

            if not osp.exists(osp.join(self.local_workplace, 'tcp_server.py')):
                url = "https://raw.githubusercontent.com/tjb-tech/agent.midware/refs/heads/main/tcp_server.py"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(osp.join(self.local_workplace, 'tcp_server.py'), 'w') as f:
                        f.write(response.text)
                    assert osp.exists(osp.join(self.local_workplace, 'tcp_server.py')), "Failed to save tcp_server.py to the local workplace"
                else:
                    raise Exception(f"Failed to download tcp_server.py from GitHub. Status code: {response.status_code}")
            if self.setup_package is not None:
                unzip_command = ["tar", "-xzvf", f"packages/{self.setup_package}.tar.gz", "-C", self.local_workplace]
                subprocess.run(unzip_command)
            if self.git_clone:
                if not os.path.exists(os.path.join(self.local_workplace, 'MetaChain')):
                    git_command = ["cd", self.local_workplace, "&&", "git", "clone", "-b", self.test_pull_name, f"https://{GITHUB_AI_TOKEN}@github.com/HKUDS/MetaChain.git"]
                    git_command = " ".join(git_command)
                    result = subprocess.run(git_command, shell=True)
                    if result.returncode != 0:
                        raise Exception(f"Failed to clone the repository. The error is: {result.stdout}")
                    copy_env_command = f"cp .env {self.local_workplace}/MetaChain"
                    result = subprocess.run(copy_env_command, shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Failed to copy .env file to the MetaChain directory. Error: {result.stderr}")
                    # create a new branch
                new_branch_name = f"{self.test_pull_name}_{self.task_name}"
                create_branch_command = f"cd {self.local_workplace}/MetaChain && git checkout -b {new_branch_name}"
                result = subprocess.run(create_branch_command, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Failed to create and switch to new branch. Error: {result.stderr}")
                    switch_branch_command = f"cd {self.local_workplace}/MetaChain && git checkout {new_branch_name}"
                    result = subprocess.run(switch_branch_command, shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Failed to switch to new branch. Error: {result.stderr}")
                    else:
                        logger.info(f"Successfully switched to new branch: {new_branch_name}")
                else:
                    logger.info(f"Successfully created and switched to new branch: {new_branch_name}")
            if existing_container.stdout.strip() == self.container_name:
                running_check_command = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
                running_container = subprocess.run(running_check_command, capture_output=True, text=True)

                if running_container.stdout.strip() == self.container_name:
                    logger.info(f"Container '{self.container_name}' is already running. Skipping creation.")
                    return
                else:
                    start_command = ["docker", "start", self.container_name]
                    subprocess.run(start_command)
                    logger.info(f"Container '{self.container_name}' has been started.")
                    return

            docker_command = [
                "docker", "run", "-d", "--name", self.container_name, "--user", "root",
                "-v", f"{self.local_workplace}:{self.docker_workplace}",
                "-w", f"{self.docker_workplace}", "-p", f"{self.communication_port}:{self.communication_port}", BASE_IMAGES,
                "/bin/bash", "-c", 
                f"python3 {self.docker_workplace}/tcp_server.py --workplace {self.workplace_name} --conda_path {self.conda_path} --port {self.communication_port}"
            ]
            result = subprocess.run(docker_command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to start container: {result.stderr}")
            logger.info(f"Container '{self.container_name}' has been created and started.")
            if self.wait_for_container_ready(timeout=60):
                logger.info(f"Container '{self.container_name}' has been created and started.")
        except Exception as e:
            logger.error(f"Docker initialization failed. Error: {e}")
            logger.error("Docker is required but not installed. Falling back to local environment.")
            self.use_emulator = True
            self.emulator.start_container()

    def stop_container(self):
        try:
            if self.use_emulator:
                self.emulator.stop_container()
                return
            stop_command = ["docker", "stop", self.container_name]
            result = subprocess.run(stop_command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to stop container: {result.stderr}")
            logger.info(f"Container '{self.container_name}' has been stopped.")
        except Exception as e:
            logger.error(f"Failed to stop container. Error: {e}")
            if self.use_emulator:
                logger.info("Using LocalDockerEmulator to stop container.")
                self.emulator.stop_container()

    def run_command(self, command, stream_callback=None):
        """communicate with docker container and execute command, support stream output"""
        if self.use_emulator:
            logger.info("Running command using LocalDockerEmulator.")
            result = self.emulator.run_command(command)
            return {
                'status': 0,
                'result': result
            }
        hostname = 'localhost'
        port = self.communication_port
        buffer_size = 4096

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((hostname, port))
            s.sendall(command.encode())

            partial_line = ""
            while True:
                chunk = s.recv(buffer_size)
                if not chunk:
                    break

                # add new received data to the unfinished data
                data = partial_line + chunk.decode('utf-8')
                lines = data.split('\n')

                # except the last line, process all complete lines
                for line in lines[:-1]:
                    if line:
                        try:
                            response = json.loads(line)
                            if response['type'] == 'chunk':
                                # process stream output
                                if stream_callback:
                                    stream_callback(response['data'])
                            elif response['type'] == 'final':
                                # return the final result
                                return {
                                    'status': response['status'],
                                    'result': response['result']
                                }
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON: {line}")

                # save the possibly unfinished last line
                partial_line = lines[-1]

        # if the loop ends normally without receiving a final response
        return {
            'status': -1,
            'result': 'Connection closed without final response'
        }

    def wait_for_container_ready(self, timeout=60):
        """Wait until the container is ready by checking if the tcp_server.py is running."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.run_command("ps aux | grep -v grep | grep tcp_server.py")
                if response['status'] == 0 and response['result'] != "":
                    return True
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to check container readiness: {e}")
        logger.error("Container did not become ready within the timeout period.")
        return False

def check_container_ports(container_name: str) -> Optional[Tuple[int, int]]:
    if not is_docker_available():
        logger.warning("Docker is not available, using local emulation.")
        return (None, None)

    container_check_command = [
        "docker", "ps", "-a",
        "--filter", f"name={container_name}",
        "--format", "{{.Ports}}"
    ]

    result = subprocess.run(container_check_command, capture_output=True, text=True)
    ports_info = result.stdout.strip()

    if not ports_info:
        return None

    # only process the mapped ports
    for mapping in ports_info.split(','):
        mapping = mapping.strip()
        if '->' in mapping:
            host_part, container_part = mapping.split('->')
            host_port = host_part.split(':')[1]  # get '12345' from '0.0.0.0:12345'
            container_port = container_part.split('/')[0]  # get '12345' from '12345/tcp'
            return (int(host_port), int(container_port))
    return None

def check_container_exist(container_name: str) -> bool:
    if not is_docker_available():
        logger.warning("Docker is not available, using local emulation.")
        return False

    container_check_command = [
        "docker", "ps", "-a",
        "--filter", f"name={container_name}",
        "--format", "{{.Names}}"
    ]
    result = subprocess.run(container_check_command, capture_output=True, text=True)
    return container_name in result.stdout.strip()

def check_container_running(container_name: str) -> bool:
    if not is_docker_available():
        logger.warning("Docker is not available, using local emulation.")
        return False

    container_check_command = [
        "docker", "ps",
        "--filter", f"name={container_name}",
        "--format", "{{.Names}}"
    ]
    result = subprocess.run(container_check_command, capture_output=True, text=True)
    return container_name in result.stdout.strip()

def with_env(env: DockerEnv):
    """将env注入到工具函数中的装饰器"""
    def decorator(func):
        def wrapped(*args, **kwargs):
            return func(env=env, *args, **kwargs)

        # 保留原始函数的所有属性
        update_wrapper(wrapped, func)
        # 修改signature，移除env参数
        wrapped.__signature__ = signature(func).replace(
            parameters=[p for p in signature(func).parameters.values() if p.name != 'env']
        )
        if func.__doc__:
            try:
                if '{docker_workplace}' in func.__doc__:
                    wrapped.__doc__ = func.__doc__.format(docker_workplace=env.docker_workplace)
                else:
                    wrapped.__doc__ = func.__doc__
                if '{local_workplace}' in func.__doc__:
                    wrapped.__doc__ = wrapped.__doc__.format(local_workplace=env.local_workplace)
                else:
                    wrapped.__doc__ = wrapped.__doc__
            except (KeyError, IndexError, ValueError):
                wrapped.__doc__ = func.__doc__
        return wrapped
    return decorator

def check_container_ports(container_name: str) -> Optional[Tuple[int, int]]:
    if not is_docker_available():
        logger.warning("Docker is not available, using local emulation.")
        return (None, None)

    container_check_command = [
        "docker", "ps", "-a",
        "--filter", f"name={container_name}",
        "--format", "{{.Ports}}"
    ]

    result = subprocess.run(container_check_command, capture_output=True, text=True)
    ports_info = result.stdout.strip()

    if not ports_info:
        return None

    # only process the mapped ports
    for mapping in ports_info.split(','):
        mapping = mapping.strip()
        if '->' in mapping:
            host_part, container_part = mapping.split('->')
            host_port = host_part.split(':')[1]  # get '12345' from '0.0.0.0:12345'
            container_port = container_part.split('/')[0]  # get '12345' from '12345/tcp'
            return (int(host_port), int(container_port))
    return None

def check_container_exist(container_name: str) -> bool:
    if not is_docker_available():
        logger.warning("Docker is not available, using local emulation.")
        return False

    container_check_command = [
        "docker", "ps", "-a",
        "--filter", f"name={container_name}",
        "--format", "{{.Names}}"
    ]
    result = subprocess.run(container_check_command, capture_output=True, text=True)
    return container_name in result.stdout.strip()

def check_container_running(container_name: str) -> bool:
    if not is_docker_available():
        logger.warning("Docker is not available, using local emulation.")
        return False

    container_check_command = [
        "docker", "ps",
        "--filter", f"name={container_name}",
        "--format", "{{.Names}}"
    ]
    result = subprocess.run(container_check_command, capture_output=True, text=True)
    return container_name in result.stdout.strip()

wd = Path(__file__).parent.resolve()