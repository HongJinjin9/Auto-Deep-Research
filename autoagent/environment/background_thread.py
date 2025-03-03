import threading
import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Generic, Tuple
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for the return type of background tasks


class TaskStatus(Enum):
    """Enum representing the status of a background task."""
    PENDING = auto()     # Task has been created but not yet started
    RUNNING = auto()     # Task is currently running
    COMPLETED = auto()   # Task completed successfully
    FAILED = auto()      # Task encountered an error
    CANCELLED = auto()   # Task was cancelled before completion


class BackgroundTask(Generic[T]):
    """
    A class to manage background tasks running in separate threads.
    
    This class provides functionality to run tasks in the background,
    monitor their status, retrieve results, and handle exceptions.
    
    Attributes:
        name (str): Name of the background task for identification
        task_function (Callable): The function to run in the background
        args (tuple): Positional arguments for the task function
        kwargs (dict): Keyword arguments for the task function
    """
    
    def __init__(self, 
                 task_function: Callable[..., T], 
                 name: str = "BackgroundTask", 
                 *args, 
                 **kwargs):
        """
        Initialize a background task.
        
        Args:
            task_function: The function to run in the background
            name: Name for this task (used for logging and identification)
            *args: Positional arguments to pass to the task function
            **kwargs: Keyword arguments to pass to the task function
        """
        self.name = name
        self.task_function = task_function
        self.args = args
        self.kwargs = kwargs
        
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._result: Optional[T] = None
        self._exception: Optional[Exception] = None
        self._status = TaskStatus.PENDING
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def start(self) -> 'BackgroundTask[T]':
        """
        Start the background task in a separate thread.
        
        Returns:
            Self reference for method chaining
            
        Raises:
            RuntimeError: If the task has already been started
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(f"Task '{self.name}' is already running")
            
            self._status = TaskStatus.RUNNING
            self._start_time = time.time()
            self._thread = threading.Thread(
                target=self._run_task_wrapper,
                name=self.name,
                daemon=True
            )
            self._thread.start()
            
        logger.debug(f"Started background task '{self.name}'")
        return self
    
    def _run_task_wrapper(self) -> None:
        """Internal method to execute the task and handle exceptions."""
        try:
            self._result = self.task_function(*self.args, **self.kwargs)
            with self._lock:
                self._status = TaskStatus.COMPLETED
        except Exception as e:
            logger.exception(f"Error in background task '{self.name}': {str(e)}")
            with self._lock:
                self._exception = e
                self._status = TaskStatus.FAILED
        finally:
            self._end_time = time.time()
    
    def is_running(self) -> bool:
        """
        Check if the task is currently running.
        
        Returns:
            bool: True if the task is running, False otherwise
        """
        with self._lock:
            return self._status == TaskStatus.RUNNING and self._thread is not None and self._thread.is_alive()
    
    def is_completed(self) -> bool:
        """
        Check if the task has completed (successfully or with error).
        
        Returns:
            bool: True if the task has completed, False otherwise
        """
        with self._lock:
            return self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the task to complete.
        
        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            bool: True if the task completed before the timeout, False otherwise
        """
        if self._thread is None:
            return False
        
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
    
    def get_result(self, timeout: Optional[float] = None, raise_exception: bool = True) -> Optional[T]:
        """
        Get the result of the task.
        
        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            raise_exception: If True, reraise any exception that occurred in the task
            
        Returns:
            The task's return value or None if not completed or failed
            
        Raises:
            Exception: If the task failed and raise_exception is True
            TimeoutError: If the task didn't complete within the timeout period
            RuntimeError: If the task hasn't been started
        """
        if self._thread is None:
            raise RuntimeError(f"Task '{self.name}' hasn't been started")
            
        if not self.wait(timeout):
            raise TimeoutError(f"Task '{self.name}' did not complete within the specified timeout")
        
        with self._lock:
            if self._exception and raise_exception:
                raise self._exception
            return self._result
    
    def cancel(self) -> bool:
        """
        Attempt to cancel the task. Note that Python doesn't support forcefully
        terminating threads, so this only marks the task as cancelled.
        
        Returns:
            bool: True if the task was successfully marked as cancelled,
                  False if the task was already completed
        """
        with self._lock:
            if self.is_completed():
                return False
                
            self._status = TaskStatus.CANCELLED
            return True
    
    def get_status(self) -> TaskStatus:
        """
        Get the current status of the task.
        
        Returns:
            TaskStatus: The current status enum value
        """
        with self._lock:
            return self._status
    
    def get_execution_time(self) -> Optional[float]:
        """
        Get the execution time of the task in seconds.
        
        Returns:
            float: The execution time if the task has completed, None otherwise
        """
        with self._lock:
            if self._start_time is None:
                return None
                
            end_time = self._end_time if self._end_time else time.time()
            return end_time - self._start_time


class BackgroundTaskManager:
    """
    A manager class to handle multiple background tasks.
    
    This allows centralized management of tasks, including creation,
    monitoring, and cleanup of background tasks.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize the background task manager.
        
        Args:
            max_workers: Maximum number of concurrent background tasks
        """
        self._tasks: Dict[str, BackgroundTask] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def create_task(self, 
                   task_function: Callable[..., T], 
                   name: Optional[str] = None, 
                   auto_start: bool = True,
                   *args, 
                   **kwargs) -> BackgroundTask[T]:
        """
        Create a new background task.
        
        Args:
            task_function: The function to run in the background
            name: Optional name for the task, defaults to function name
            auto_start: Whether to start the task immediately
            *args: Positional arguments for the task function
            **kwargs: Keyword arguments for the task function
            
        Returns:
            The created background task
            
        Raises:
            ValueError: If a task with the same name already exists
        """
        if name is None:
            name = f"{task_function.__name__}_{id(task_function)}"
            
        with self._lock:
            if name in self._tasks:
                raise ValueError(f"Task with name '{name}' already exists")
                
            task = BackgroundTask(task_function, name, *args, **kwargs)
            self._tasks[name] = task
            
            if auto_start:
                task.start()
                
            return task
    
    def get_task(self, name: str) -> Optional[BackgroundTask]:
        """
        Get a task by name.
        
        Args:
            name: The name of the task to retrieve
            
        Returns:
            The task object if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(name)
    
    def list_tasks(self) -> Dict[str, TaskStatus]:
        """
        Get a dictionary of all managed tasks with their current status.
        
        Returns:
            Dict mapping task names to their status
        """
        with self._lock:
            return {name: task.get_status() for name, task in self._tasks.items()}
    
    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            bool: True if all tasks completed, False otherwise
        """
        start_time = time.time()
        remaining_time = timeout
        
        with self._lock:
            tasks = list(self._tasks.values())
        
        for task in tasks:
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_time = max(0, timeout - elapsed)
                if remaining_time <= 0:
                    return False
                
            if not task.wait(remaining_time):
                return False
                
        return True
    
    def cancel_all(self) -> None:
        """Cancel all managed tasks."""
        with self._lock:
            for task in self._tasks.values():
                task.cancel()
    
    def remove_completed(self) -> int:
        """
        Remove completed tasks from the manager.
        
        Returns:
            int: Number of tasks removed
        """
        to_remove = []
        
        with self._lock:
            for name, task in self._tasks.items():
                if task.is_completed():
                    to_remove.append(name)
                    
            for name in to_remove:
                del self._tasks[name]
                
        return len(to_remove)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the task manager and cleanup resources.
        
        Args:
            wait: If True, wait for tasks to complete before shutting down
        """
        if wait:
            self.wait_all()
        else:
            self.cancel_all()
            
        self._executor.shutdown(wait=wait)
        
        with self._lock:
            self._tasks.clear()


# Utility functions for working with async code in background threads
def run_async_in_thread(coroutine) -> BackgroundTask:
    """
    Run an async coroutine in a background thread with its own event loop.
    
    Args:
        coroutine: The coroutine to run
        
    Returns:
        BackgroundTask: A background task that will contain the result
    """
    def run_coroutine():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()
    
    return BackgroundTask(run_coroutine, name=f"AsyncTask_{id(coroutine)}").start()