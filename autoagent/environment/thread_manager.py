import threading
import logging
import time
import uuid
import queue
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from contextlib import contextmanager
import traceback

from autoagent.logger import LoggerManager

# Type variable for the return type of thread functions
T = TypeVar('T')

# Create a dedicated logger for thread management
logger = logging.getLogger("thread_manager")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Message priority levels
class MessagePriority(Enum):
    """Enum representing message priority levels for thread communication."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ThreadStatus(Enum):
    """Enum representing the status of a thread."""
    PENDING = auto()     # Thread has been created but not yet started
    RUNNING = auto()     # Thread is currently running
    COMPLETED = auto()   # Thread completed successfully
    FAILED = auto()      # Thread encountered an error
    CANCELLED = auto()   # Thread was cancelled before completion


@dataclass
class ThreadMessage:
    """Class representing a message exchanged between threads."""
    content: Any
    sender_id: str
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __lt__(self, other):
        # For priority queue ordering - higher priority messages come first
        if not isinstance(other, ThreadMessage):
            return NotImplemented
        return self.priority.value > other.priority.value


class NewThread(Generic[T]):
    """
    A class to manage threads created from a root thread with enhanced 
    functionality for lifecycle management and inter-thread communication.
    
    Attributes:
        thread_id (str): Unique identifier for the thread
        name (str): Human-readable name for the thread
        status (ThreadStatus): Current status of the thread
        parent_thread_id (str): ID of the parent thread
    """
    
    def __init__(
        self,
        target_func: Callable[..., T],
        name: str = None,
        parent_thread_id: str = None,
        args: tuple = (),
        kwargs: dict = None,
        auto_recover: bool = False,
        max_recovery_attempts: int = 3,
        recovery_delay: float = 1.0
    ):
        """
        Initialize a new managed thread.
        
        Args:
            target_func: Function to execute in the thread
            name: Human-readable name (defaults to function name + UUID)
            parent_thread_id: ID of the parent thread that created this thread
            args: Positional arguments to pass to target_func
            kwargs: Keyword arguments to pass to target_func
            auto_recover: Whether to automatically restart the thread on failure
            max_recovery_attempts: Maximum number of recovery attempts
            recovery_delay: Delay in seconds between recovery attempts
        """
        self.target_func = target_func
        self.thread_id = str(uuid.uuid4())
        self.name = name or f"{target_func.__name__}_{self.thread_id[:8]}"
        self.parent_thread_id = parent_thread_id
        self.args = args
        self.kwargs = kwargs or {}
        
        # Thread state
        self._status = ThreadStatus.PENDING
        self._thread = None
        self._stop_event = threading.Event()
        self._result = None
        self._exception = None
        self._lock = threading.RLock()
        
        # Timing and statistics
        self._start_time = None
        self._end_time = None
        
        # Auto-recovery settings
        self.auto_recover = auto_recover
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_delay = recovery_delay
        self._recovery_attempts = 0
        
        # Message queue for inter-thread communication
        self._message_queue = queue.PriorityQueue()
        self._message_callbacks = set()
        
        # Shared data storage
        self._shared_data = {}
        self._data_lock = threading.RLock()
        
        logger.debug(f"Thread '{self.name}' (ID: {self.thread_id}) initialized")
    
    def start(self):
        """
        Start the thread execution.
        
        Returns:
            self: For method chaining
            
        Raises:
            RuntimeError: If the thread is already running
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning(f"Thread '{self.name}' is already running")
                raise RuntimeError(f"Thread '{self.name}' is already running")
            
            self._status = ThreadStatus.RUNNING
            self._start_time = time.time()
            self._thread = threading.Thread(
                target=self._execute_with_safety,
                name=self.name,
                daemon=True  # Allow the program to exit even if this thread is running
            )
            self._thread.start()
            
            logger.info(f"Thread '{self.name}' (ID: {self.thread_id}) started")
            
        return self
    
    def _execute_with_safety(self):
        """
        Wrapper to execute the target function with exception handling and state management.
        """
        try:
            logger.debug(f"Thread '{self.name}' executing target function")
            self._result = self.target_func(*self.args, **self.kwargs)
            with self._lock:
                self._status = ThreadStatus.COMPLETED
                logger.info(f"Thread '{self.name}' completed successfully")
        except Exception as e:
            logger.error(f"Error in thread '{self.name}': {str(e)}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            with self._lock:
                self._exception = e
                self._status = ThreadStatus.FAILED
            
            # Handle automatic recovery if enabled
            if self.auto_recover and self._recovery_attempts < self.max_recovery_attempts:
                self._attempt_recovery()
        finally:
            self._end_time = time.time()
            duration = self._end_time - (self._start_time or self._end_time)
            logger.debug(f"Thread '{self.name}' executed for {duration:.2f} seconds")
    
    def _attempt_recovery(self):
        """Attempt to recover a failed thread by restarting it."""
        self._recovery_attempts += 1
        logger.warning(
            f"Attempting recovery #{self._recovery_attempts} for thread '{self.name}' "
            f"in {self.recovery_delay:.2f} seconds"
        )
        
        time.sleep(self.recovery_delay)
        
        with self._lock:
            if self._status != ThreadStatus.CANCELLED:
                self._status = ThreadStatus.PENDING
                self._start()
    
    def stop(self, timeout: float = None, join: bool = True):
        """
        Signal the thread to stop execution.
        
        Args:
            timeout: Maximum time to wait for thread termination if join=True
            join: Whether to join the thread after setting the stop event
            
        Returns:
            bool: True if the thread was successfully stopped or already completed
        """
        self._stop_event.set()
        
        with self._lock:
            if self._status in (ThreadStatus.COMPLETED, ThreadStatus.FAILED, ThreadStatus.CANCELLED):
                # Thread is already stopped
                return True
                
            logger.info(f"Stopping thread '{self.name}'")
            self._status = ThreadStatus.CANCELLED
        
        if join and self._thread is not None:
            self._thread.join(timeout=timeout)
            was_joined = not self._thread.is_alive()
            if not was_joined:
                logger.warning(f"Thread '{self.name}' did not terminate within timeout ({timeout}s)")
            return was_joined
        
        return True
    
    def join(self, timeout: float = None):
        """
        Wait for the thread to finish execution.
        
        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            bool: True if the thread completed before the timeout, False otherwise
        """
        if self._thread is None:
            return False
        
        self._thread.join(timeout=timeout)
        thread_finished = not self._thread.is_alive()
        
        if thread_finished:
            logger.debug(f"Successfully joined thread '{self.name}'")
        else:
            logger.debug(f"Timeout reached while joining thread '{self.name}'")
        
        return thread_finished
    
    def is_alive(self):
        """
        Check if the thread is currently running.
        
        Returns:
            bool: True if the thread is running, False otherwise
        """
        return self._thread is not None and self._thread.is_alive()
    
    def is_stopped(self):
        """
        Check if the thread has been signaled to stop.
        
        Returns:
            bool: True if the stop event is set
        """
        return self._stop_event.is_set()
    
    def _update_status(self, new_status: ThreadStatus):
        """
        Update the thread status with proper locking and logging.
        
        Args:
            new_status: The new thread status to set
        """
        with self._lock:
            old_status = self._status
            self._status = new_status
            
            if old_status != new_status:
                logger.info(f"Thread '{self.name}' status changed: {old_status} → {new_status}")
    
    def get_status(self) -> ThreadStatus:
        """
        Get the current thread status.
        
        Returns:
            ThreadStatus: The current status of the thread
        """
        with self._lock:
            return self._status
    
    def get_result(self, timeout: float = None, raise_exception: bool = True) -> Optional[T]:
        """
        Get the result of the thread execution.
        
        Args:
            timeout: Maximum time to wait for the thread to complete
            raise_exception: Whether to re-raise any exception that occurred during execution
            
        Returns:
            The thread's return value or None if not completed or failed
            
        Raises:
            Exception: If the thread failed and raise_exception is True
            TimeoutError: If the thread didn't complete within the timeout period
            RuntimeError: If the thread hasn't been started
        """
        if self._thread is None:
            raise RuntimeError(f"Thread '{self.name}' hasn't been started")
            
        if not self.join(timeout=timeout):
            raise TimeoutError(f"Thread '{self.name}' did not complete within the timeout")
        
        with self._lock:
            if self._exception and raise_exception:
                logger.debug(f"Re-raising exception from thread '{self.name}'")
                raise self._exception
            return self._result
    
    def get_execution_time(self) -> Optional[float]:
        """
        Get the thread execution time in seconds.
        
        Returns:
            float: The execution time if started, None otherwise
        """
        if self._start_time is None:
            return None
        
        end_time = self._end_time if self._end_time else time.time()
        return end_time - self._start_time
    
    def get_exception(self) -> Optional[Exception]:
        """
        Get any exception that occurred during thread execution.
        
        Returns:
            Exception or None if no exception occurred
        """
        with self._lock:
            return self._exception
    
    # Thread communication methods
    def send_message(self, content: Any, priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """
        Send a message to this thread.
        
        Args:
            content: Message content (any picklable object)
            priority: Priority level of the message
            
        Returns:
            str: The message ID
        """
        # Get the current thread ID as the sender
        current_thread_id = threading.current_thread().ident
        
        message = ThreadMessage(
            content=content,
            sender_id=str(current_thread_id),
            priority=priority
        )
        
        self._message_queue.put(message)
        logger.debug(f"Message sent to thread '{self.name}': {message.message_id}")
        
        return message.message_id
    
    def receive_message(self, timeout: float = None) -> Optional[ThreadMessage]:
        """
        Receive a message from the thread's message queue.
        
        Args:
            timeout: Maximum time to wait for a message, or None to wait indefinitely
            
        Returns:
            ThreadMessage or None if timeout occurs
        """
        try:
            message = self._message_queue.get(block=True, timeout=timeout)
            self._message_queue.task_done()
            
            # Trigger any registered callbacks
            for callback in self._message_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {str(e)}")
            
            logger.debug(f"Thread '{self.name}' received message: {message.message_id}")
            return message
        except queue.Empty:
            return None
    
    def register_callback(self, callback_func: Callable[[ThreadMessage], None]) -> None:
        """
        Register a callback function to be called when a message is received.
        
        Args:
            callback_func: Function that takes a ThreadMessage as its parameter
        """
        self._message_callbacks.add(callback_func)
        logger.debug(f"Callback registered for thread '{self.name}'")
    
    def unregister_callback(self, callback_func: Callable[[ThreadMessage], None]) -> bool:
        """
        Unregister a previously registered callback function.
        
        Args:
            callback_func: The callback function to remove
            
        Returns:
            bool: True if the callback was found and removed, False otherwise
        """
        if callback_func in self._message_callbacks:
            self._message_callbacks.remove(callback_func)
            logger.debug(f"Callback unregistered from thread '{self.name}'")
            return True
        return False
    
    # Shared data methods
    def set_shared_data(self, key: str, value: Any) -> None:
        """
        Set a value in the thread's shared data store.
        
        Args:
            key: Data key
            value: Data value (must be picklable)
        """
        with self._data_lock:
            self._shared_data[key] = value
            logger.debug(f"Thread '{self.name}' set shared data: {key}")
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the thread's shared data store.
        
        Args:
            key: Data key
            default: Default value to return if key doesn't exist
            
        Returns:
            The stored value or the default
        """
        with self._data_lock:
            return self._shared_data.get(key, default)
    
    def delete_shared_data(self, key: str) -> bool:
        """
        Delete a key from the thread's shared data store.
        
        Args:
            key: Data key to delete
            
        Returns:
            bool: True if the key existed and was deleted
        """
        with self._data_lock:
            if key in self._shared_data:
                del self._shared_data[key]
                logger.debug(f"Thread '{self.name}' deleted shared data: {key}")
                return True
            return False


class ThreadManager:
    """
    A manager class to handle multiple NewThread instances.
    
    This class provides functionality to create, monitor, and control
    threads in a centralized way, particularly when created from a root thread.
    """
    
    def __init__(self):
        """Initialize the thread manager."""
        self._threads: Dict[str, NewThread] = {}
        self._lock = threading.RLock()
        self._root_thread_id = str(threading.current_thread().ident)
        self._monitoring_enabled = False
        self._monitor_interval = 5.0  # seconds
        self._monitor_thread = None
        
        logger.info(f"ThreadManager initialized with root thread ID: {self._root_thread_id}")
    
    def create_thread(self, 
                     target_func: Callable[..., T],
                     name: str = None,
                     auto_start: bool = True,
                     auto_recover: bool = False,
                     max_recovery_attempts: int = 3,
                     recovery_delay: float = 1.0,
                     *args,
                     **kwargs) -> NewThread[T]:
        """
        Create a new managed thread.
        
        Args:
            target_func: Function to execute in the thread
            name: Human-readable name for the thread
            auto_start: Whether to start the thread immediately
            auto_recover: Whether to automatically restart the thread on failure
            max_recovery_attempts: Maximum number of recovery attempts
            recovery_delay: Delay in seconds between recovery attempts
            *args: Positional arguments for the target function
            **kwargs: Keyword arguments for the target function
            
        Returns:
            The created NewThread instance
            
        Raises:
            ValueError: If a thread with the same name already exists
        """
        if name is None:
            name = f"{target_func.__name__}_{uuid.uuid4().hex[:8]}"
            
        with self._lock:
            if name in self._threads:
                logger.error(f"Thread with name '{name}' already exists")
                raise ValueError(f"Thread with name '{name}' already exists")
                
            thread = NewThread(
                target_func=target_func,
                name=name,
                parent_thread_id=self._root_thread_id,
                args=args,
                kwargs=kwargs,
                auto_recover=auto_recover,
                max_recovery_attempts=max_recovery_attempts,
                recovery_delay=recovery_delay
            )
            
            self._threads[thread.thread_id] = thread
            logger.info(f"Thread '{name}' (ID: {thread.thread_id}) created")
            
            if auto_start:
                thread.start()
                
            return thread
    
    def get_thread(self, thread_id: str) -> Optional[NewThread]:
        """
        Get a thread by its ID.
        
        Args:
            thread_id: The ID of the thread to retrieve
            
        Returns:
            The thread object if found, None otherwise
        """
        with self._lock:
            return self._threads.get(thread_id)
    
    def get_thread_by_name(self, name: str) -> Optional[NewThread]:
        """
        Get a thread by its name.
        
        Args:
            name: The name of the thread to retrieve
            
        Returns:
            The first thread object with the given name, or None if not found
        """
        with self._lock:
            for thread in self._threads.values():
                if thread.name == name:
                    return thread
            return None
    
    def get_all_threads(self) -> Dict[str, NewThread]:
        """
        Get all managed threads.
        
        Returns:
            Dictionary mapping thread IDs to thread objects
        """
        with self._lock:
            return self._threads.copy()
    
    def get_thread_statuses(self) -> Dict[str, ThreadStatus]:
        """
        Get the status of all managed threads.
        
        Returns:
            Dictionary mapping thread IDs to their current status
        """
        with self._lock:
            return {tid: thread.get_status() for tid, thread in self._threads.items()}
    
    def start_all(self) -> None:
        """Start all pending threads."""
        with self._lock:
            for thread in self._threads.values():
                if thread.get_status() == ThreadStatus.PENDING:
                    try:
                        thread.start()
                    except RuntimeError as e:
                        logger.error(f"Failed to start thread '{thread.name}': {str(e)}")
    
    def stop_all(self, timeout: float = None, join: bool = True) -> None:
        """
        Stop all running threads.
        
        Args:
            timeout: Maximum time to wait for each thread to stop
            join: Whether to wait for threads to finish
        """
        with self._lock:
            for thread in self._threads.values():
                if thread.get_status() == ThreadStatus.RUNNING:
                    thread.stop(timeout=timeout, join=join)
    
    def join_all(self, timeout: float = None) -> bool:
        """
        Wait for all threads to complete.
        
        Args:
            timeout: Maximum time to wait for all threads to complete
            
        Returns:
            bool: True if all threads completed, False if timeout occurred
        """
        if timeout is not None:
            deadline = time.time() + timeout
        
        all_joined = True
        
        with self._lock:
            for thread in self._threads.values():
                if timeout is not None:
                    remaining = max(0, deadline - time.time())
                    if not thread.join(timeout=remaining):
                        all_joined = False
                else:
                    thread.join()
        
        return all_joined
    
    def broadcast_message(self, content: Any, 
                          priority: MessagePriority = MessagePriority.NORMAL,
                          exclude_ids: Set[str] = None) -> Dict[str, str]:
        """
        Broadcast a message to all managed threads.
        
        Args:
            content: Message content
            priority: Message priority level
            exclude_ids: Set of thread IDs to exclude from broadcast
            
        Returns:
            Dictionary mapping thread IDs to message IDs
        """
        exclude_ids = exclude_ids or set()
        message_ids = {}
        
        with self._lock:
            for tid, thread in self._threads.items():
                if tid not in exclude_ids:
                    message_id = thread.send_message(content, priority)
                    message_ids[tid] = message_id
        
        logger.debug(f"Broadcast message sent to {len(message_ids)} threads")
        return message_ids
    
    def remove_thread(self, thread_id: str) -> bool:
        """
        Remove a thread from management.
        
        Args:
            thread_id: ID of the thread to remove
            
        Returns:
            bool: True if the thread was found and removed
        """
        with self._lock:
            if thread_id in self._threads:
                thread = self._threads[thread_id]
                if thread.is_alive():
                    logger.warning(f"Removing thread '{thread.name}' while still running")
                del self._threads[thread_id]
                logger.info(f"Thread '{thread.name}' (ID: {thread_id}) removed from manager")
                return True
            return False
    
    def cleanup_completed(self) -> int:
        """
        Remove all completed or failed threads from management.
        
        Returns:
            int: Number of threads cleaned up
        """
        to_remove = []
        
        with self._lock:
            for tid, thread in self._threads.items():
                status = thread.get_status()
                if status in (ThreadStatus.COMPLETED, ThreadStatus.FAILED, ThreadStatus.CANCELLED):
                    to_remove.append(tid)
            
            for tid in to_remove:
                del self._threads[tid]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed/failed threads")
        
        return len(to_remove)
    
    def start_monitoring(self, interval: float = 5.0) -> bool:
        """
        Start a background thread to monitor thread health.
        
        Args:
            interval: Time in seconds between health checks
            
        Returns:
            bool: True if monitoring started, False if already running
        """
        with self._lock:
            if self._monitoring_enabled:
                logger.warning("Thread monitoring is already enabled")
                return False
            
            self._monitoring_enabled = True
            self._monitor_interval = interval
            
            self._monitor_thread = threading.Thread(
                target=self._monitor_threads,
                name="ThreadMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            logger.info(f"Thread monitoring started with interval {interval} seconds")
            return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop the thread monitoring.
        
        Returns:
            bool: True if monitoring was stopped, False if not running
        """
        with self._lock:
            if not self._monitoring_enabled:
                logger.warning("Thread monitoring is not enabled")
                return False
            
            self._monitoring_enabled = False
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=self._monitor_interval + 1)
            
            logger.info("Thread monitoring stopped")
            return True
    
    def _monitor_threads(self) -> None:
        """Background task to monitor thread health."""
        while self._monitoring_enabled:
            try:
                self._check_thread_health()
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in thread monitor: {str(e)}")
    
    def _check_thread_health(self) -> None:
        """Check the health of all managed threads."""
        issues = 0
        with self._lock:
            for thread in self._threads.values():
                status = thread.get_status()
                
                # Check for hung threads
                if status == ThreadStatus.RUNNING:
                    runtime = thread.get_execution_time()
                    if runtime is not None and runtime > 300:  # 5 minutes
                        logger.warning(f"Thread '{thread.name}' has been running for {runtime:.1f} seconds")
                        issues += 1
                
                # Check for failed threads
                if status == ThreadStatus.FAILED:
                    exception = thread.get_exception()
                    exception_msg = str(exception) if exception else "Unknown error"
                    logger.error(f"Thread '{thread.name}' has failed with error: {exception_msg}")
                    issues += 1
            
            if issues > 0:
                logger.warning(f"Found {issues} thread(s) with potential issues")
    
    @contextmanager
    def run_in_thread(self, name: str = None, auto_recover: bool = False) -> NewThread:
        """
        Context manager to run a block of code in a new thread.
        
        Args:
            name: Name for the thread
            auto_recover: Whether to auto-restart on failure
            
        Yields:
            NewThread: The thread object
        """
        def wrapper():
            # This will be filled in by the context manager
            return self._thread_context_func(*self._thread_context_args, 
                                           **self._thread_context_kwargs)
        
        thread = self.create_thread(
            target_func=wrapper,
            name=name,
            auto_start=False,
            auto_recover=auto_recover
        )
        
        try:
            self._thread_context_func = None
            self._thread_context_args = ()
            self._thread_context_kwargs = {}
            yield thread
            
            # The actual function is assigned by the user in the context
            if self._thread_context_func is not None:
                thread.start()
        finally:
            # Clean up context variables
            self._thread_context_func = None
            self._thread_context_args = None
            self._thread_context_kwargs = None
    
    def execute_in_thread(self, func: Callable, *args, **kwargs) -> NewThread:
        """
        Helper method to be used with run_in_thread context manager.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            self for chaining
        """
        self._thread_context_func = func
        self._thread_context_args = args
        self._thread_context_kwargs = kwargs
        return self
    
    def shutdown(self, wait: bool = True, timeout: float = None) -> None:
        """
        Shutdown the thread manager, stopping all threads.
        
        Args:
            wait: Whether to wait for threads to finish
            timeout: Maximum time to wait for each thread
        """
        # Stop monitoring first
        if self._monitoring_enabled:
            self.stop_monitoring()
        
        # Stop all threads
        self.stop_all(timeout=timeout, join=wait)
        
        # Clear threads dictionary
        with self._lock:
            self._threads.clear()
            
        logger.info("ThreadManager shut down")