from autoagent.environment.docker_env import DockerEnv, DockerConfig
from typing import Optional

import tiktoken
from datetime import datetime

def get_current_time() -> str:
    """
    Get the current time formatted as a string.
    
    Returns:
        A string representation of the current time in format YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def truncate_by_tokens(env: DockerEnv, text: str, max_tokens: int = 4096, model: str = "gpt-4o-2024-08-06") -> str:
    """
    Truncate text to stay within token limits, saving the full text to a file if needed.
    
    Args:
        env: Docker environment instance
        text: The text to potentially truncate
        max_tokens: Maximum number of tokens allowed
        model: The model to use for tokenization
        
    Returns:
        Truncated text if necessary, with information about where the full text is saved
    """
    from autoagent.tools.terminal_tools import create_file
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = f"{env.docker_workplace}/console_output/truncated_output_{timestamp}.txt"
    create_msg = create_file(output_path, content=text, context_variables={'code_env': env})
    # 截断tokens并解码回字符串
    truncated_tokens_bos = tokens[:max_tokens//2]
    truncated_tokens_eos = tokens[-(max_tokens - len(truncated_tokens_bos)):]
    if create_msg.startswith("Error"):
        return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full console output is too long, so I want to save them into the file: {0}\n\nBut I got an error: {1}".format(output_path, create_msg)
    
    return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full console output is too long, so it is saved in the file: {0}\n\nYou may use the `File Surfer Agent` to view the full output.".format(output_path)