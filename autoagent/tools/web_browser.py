import logging
from typing import Optional, Dict, Any, List
from autoagent.registry import register_tool
from autoagent.environment.browser_env import BrowserEnv
from autoagent.types import Result

logger = logging.getLogger(__name__)

@register_tool(name="click")
def click(
    selector: str,
    env: Optional[BrowserEnv] = None,
    wait_time: float = 1.0,
    max_retries: int = 3,
    position: Optional[Dict[str, float]] = None,
    double_click: bool = False,
    right_click: bool = False,
    check_visible: bool = True,
    timeout: float = 30.0
) -> Result:
    """
    Simulate clicking on an element in a web page.
    
    Args:
        selector: CSS selector or XPath for the element to click
        env: BrowserEnv instance to use (will be injected if not provided)
        wait_time: Time to wait after clicking in seconds
        max_retries: Maximum number of retries if the click fails
        position: Optional position within element to click, e.g. {"x": 0.5, "y": 0.5} for center
        double_click: Whether to perform a double click
        right_click: Whether to perform a right click
        check_visible: Whether to check if the element is visible before clicking
        timeout: Maximum time to wait for the element to be available
        
    Returns:
        Result object with success status and message
    """
    if env is None:
        logger.error("No browser environment provided")
        return Result(success=False, message="No browser environment provided")
    
    try:
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Check if the element exists and is accessible
                if check_visible:
                    visible = env.page.is_visible(selector, timeout=timeout/2)
                    if not visible:
                        logger.warning(f"Element {selector} is not visible, retrying...")
                        retry_count += 1
                        continue
                
                # Apply different click types
                if double_click:
                    env.page.dblclick(selector, position=position, timeout=timeout)
                    logger.info(f"Double-clicked element {selector}")
                elif right_click:
                    env.page.click(selector, button="right", position=position, timeout=timeout)
                    logger.info(f"Right-clicked element {selector}")
                else:
                    env.page.click(selector, position=position, timeout=timeout)
                    logger.info(f"Clicked element {selector}")
                
                # Wait after clicking
                if wait_time > 0:
                    env.page.wait_for_timeout(wait_time * 1000)  # convert to milliseconds
                
                return Result(success=True, message=f"Successfully clicked on {selector}")
            
            except Exception as e:
                logger.warning(f"Click attempt {retry_count+1} failed: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                
                # Wait briefly before retrying
                env.page.wait_for_timeout(1000)
    
    except Exception as e:
        error_msg = f"Failed to click element {selector} after {max_retries} attempts: {str(e)}"
        logger.error(error_msg)
        return Result(success=False, message=error_msg)

@register_tool(name="hover")
def hover(
    selector: str,
    env: Optional[BrowserEnv] = None,
    wait_time: float = 1.0,
    timeout: float = 10.0
) -> Result:
    """
    Hover over an element on a web page.
    
    Args:
        selector: CSS selector or XPath for the element to hover over
        env: BrowserEnv instance to use (will be injected if not provided)
        wait_time: Time to wait after hovering in seconds
        timeout: Maximum time to wait for the element to be available
        
    Returns:
        Result object with success status and message
    """
    if env is None:
        logger.error("No browser environment provided")
        return Result(success=False, message="No browser environment provided")
    
    try:
        env.page.hover(selector, timeout=timeout * 1000)
        
        # Wait after hovering
        if wait_time > 0:
            env.page.wait_for_timeout(wait_time * 1000)
        
        return Result(success=True, message=f"Successfully hovered over {selector}")
    
    except Exception as e:
        error_msg = f"Failed to hover over element {selector}: {str(e)}"
        logger.error(error_msg)
        return Result(success=False, message=error_msg)

@register_tool(name="fill_form")
def fill_form(
    selector: str,
    value: str,
    env: Optional[BrowserEnv] = None,
    wait_time: float = 0.5,
    clear_first: bool = True,
    timeout: float = 10.0
) -> Result:
    """
    Fill a form field on a web page.
    
    Args:
        selector: CSS selector or XPath for the form element
        value: Text to input into the form field
        env: BrowserEnv instance to use (will be injected if not provided)
        wait_time: Time to wait after filling in seconds
        clear_first: Whether to clear the field before filling
        timeout: Maximum time to wait for the element to be available
        
    Returns:
        Result object with success status and message
    """
    if env is None:
        logger.error("No browser environment provided")
        return Result(success=False, message="No browser environment provided")
    
    try:
        if clear_first:
            env.page.fill(selector, "", timeout=timeout * 1000)
        
        env.page.fill(selector, value, timeout=timeout * 1000)
        
        # Wait after filling
        if wait_time > 0:
            env.page.wait_for_timeout(wait_time * 1000)
        
        return Result(success=True, message=f"Successfully filled {selector} with text")
    
    except Exception as e:
        error_msg = f"Failed to fill form element {selector}: {str(e)}"
        logger.error(error_msg)
        return Result(success=False, message=error_msg)