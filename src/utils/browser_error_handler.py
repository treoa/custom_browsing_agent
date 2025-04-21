"""
Browser Error Handler

This module provides utilities for handling browser errors and debugging
in the context of research agents using browser-use.
"""

import os
import time
import asyncio
import traceback
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

from .project_logger import ProjectLogger


class BrowserErrorHandler:
    """
    Utility class for handling browser errors and debugging browser-related issues.
    
    This class provides methods for logging browser errors, capturing debugging
    information, and implementing recovery strategies for browser failures.
    """
    
    def __init__(
        self,
        logger: Optional[ProjectLogger] = None,
        project_dir: Optional[str] = None,
        max_retry_attempts: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize a BrowserErrorHandler instance.
        
        Args:
            logger: Optional project logger
            project_dir: Directory to store error logs and screenshots
            max_retry_attempts: Maximum retry attempts for browser actions
            retry_delay: Delay in seconds between retry attempts
        """
        self.logger = logger or logging.getLogger("browser_error_handler")
        self.project_dir = project_dir
        
        # Set up error log directory if project directory is provided
        if project_dir:
            self.error_dir = os.path.join(project_dir, "logs", "errors")
            os.makedirs(self.error_dir, exist_ok=True)
        else:
            self.error_dir = None
        
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        
        # Error tracking
        self.error_count = 0
        self.errors = []
        self.last_error_time = 0
    
    async def handle_browser_error(
        self,
        error: Exception,
        browser_context: Optional[BrowserContext] = None,
        action: Optional[Dict[str, Any]] = None,
        retry_action: Optional[callable] = None
    ) -> Tuple[bool, Optional[ActionResult]]:
        """
        Handle a browser error with logging and recovery attempts.
        
        Args:
            error: The exception that occurred
            browser_context: Optional browser context for debugging
            action: Optional action that caused the error
            retry_action: Optional function to retry the action
            
        Returns:
            Tuple of (success, result) where success is a boolean and result is the ActionResult
        """
        # Track error
        self.error_count += 1
        self.last_error_time = time.time()
        
        # Prepare error info
        error_info = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
            "action": action
        }
        
        # Log the error
        error_id = f"browser_error_{int(self.last_error_time)}"
        self.logger.error(
            f"Browser error: {error}",
            extra_context={
                "error_id": error_id,
                "error_type": error.__class__.__name__,
                "action": str(action)
            },
            exc_info=True
        )
        
        # Save detailed error information to file
        await self._save_error_info(error_id, error_info, browser_context)
        
        # Add to error list
        self.errors.append(error_info)
        
        # Determine if we should retry
        if retry_action is not None and self.error_count <= self.max_retry_attempts:
            self.logger.info(
                f"Retrying action (attempt {self.error_count}/{self.max_retry_attempts})"
            )
            
            # Delay before retry
            await asyncio.sleep(self.retry_delay)
            
            try:
                # Retry the action
                result = await retry_action()
                
                # Log success
                self.logger.info(
                    f"Retry successful on attempt {self.error_count}"
                )
                
                return True, result
            except Exception as retry_error:
                # Log retry failure
                self.logger.error(
                    f"Retry failed: {retry_error}",
                    extra_context={"error_id": error_id, "retry_attempt": self.error_count},
                    exc_info=True
                )
        
        # Create error result
        error_result = ActionResult(
            error=f"Browser action failed: {str(error)}",
            include_in_memory=True
        )
        
        return False, error_result
    
    async def _save_error_info(
        self,
        error_id: str,
        error_info: Dict[str, Any],
        browser_context: Optional[BrowserContext] = None
    ) -> None:
        """
        Save detailed error information to files.
        
        Args:
            error_id: Unique identifier for the error
            error_info: Dictionary with error information
            browser_context: Optional browser context for taking screenshot
        """
        if not self.error_dir:
            return
        
        # Save error info to JSON file
        error_file = os.path.join(self.error_dir, f"{error_id}.json")
        try:
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save error info: {e}")
        
        # Take screenshot if browser context is provided
        if browser_context:
            try:
                # Get current page
                page = await browser_context.get_current_page()
                
                # Take screenshot
                screenshot_path = os.path.join(self.error_dir, f"{error_id}.png")
                await page.screenshot(path=screenshot_path)
                
                self.logger.info(f"Saved error screenshot to {screenshot_path}")
                
                # Save page content
                html_path = os.path.join(self.error_dir, f"{error_id}.html")
                content = await page.content()
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                self.logger.info(f"Saved page content to {html_path}")
                
            except Exception as screenshot_error:
                self.logger.error(f"Failed to capture error screenshot: {screenshot_error}")
    
    async def check_browser_health(
        self,
        browser_context: BrowserContext
    ) -> bool:
        """
        Check the health of the browser context.
        
        Args:
            browser_context: Browser context to check
            
        Returns:
            True if browser is healthy, False otherwise
        """
        try:
            # Try to get browser state
            state = await browser_context.get_state()
            
            # Try to access current page
            page = await browser_context.get_current_page()
            
            # Try a simple evaluation
            await page.evaluate("1 + 1")
            
            return True
        except Exception as e:
            self.logger.error(f"Browser health check failed: {e}")
            return False
    
    async def recover_browser_context(
        self,
        browser_context: BrowserContext,
        url: Optional[str] = "about:blank"
    ) -> bool:
        """
        Attempt to recover a browser context by navigating to a simple page.
        
        Args:
            browser_context: Browser context to recover
            url: URL to navigate to for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            # Get current page or create new one
            try:
                page = await browser_context.get_current_page()
            except Exception:
                # If getting current page fails, try to create a new one
                page = await browser_context.new_page()
            
            # Navigate to a simple page
            await page.goto(url)
            
            # Verify browser is responsive
            await page.evaluate("1 + 1")
            
            self.logger.info("Browser context recovered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to recover browser context: {e}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about browser errors.
        
        Returns:
            Dictionary with error statistics
        """
        # Count errors by type
        error_types = {}
        for error in self.errors:
            error_type = error.get("error_type", "Unknown")
            if error_type in error_types:
                error_types[error_type] += 1
            else:
                error_types[error_type] = 1
        
        return {
            "total_errors": self.error_count,
            "error_types": error_types,
            "last_error_time": self.last_error_time,
            "has_recent_errors": time.time() - self.last_error_time < 60  # Errors in the last minute
        }
    
    async def create_error_report(self) -> str:
        """
        Create a comprehensive error report.
        
        Returns:
            Path to the generated error report file
        """
        if not self.error_dir:
            return "No error directory configured"
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.error_dir, f"error_report_{timestamp}.md")
        
        # Generate report content
        stats = self.get_error_statistics()
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Browser Error Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Error Statistics\n\n")
            f.write(f"Total Errors: {stats['total_errors']}\n\n")
            
            if stats['error_types']:
                f.write("Error Types:\n\n")
                for error_type, count in stats['error_types'].items():
                    f.write(f"- {error_type}: {count}\n")
            
            f.write("\n## Recent Errors\n\n")
            
            # Include the 10 most recent errors
            recent_errors = sorted(self.errors, key=lambda e: e.get("timestamp", ""), reverse=True)[:10]
            
            for i, error in enumerate(recent_errors):
                f.write(f"### Error {i+1}: {error.get('error_type', 'Unknown')}\n\n")
                f.write(f"Time: {error.get('timestamp', 'Unknown')}\n\n")
                f.write(f"Message: {error.get('error_message', 'No message')}\n\n")
                
                # Include action if available
                if error.get("action"):
                    f.write("Action:\n```json\n")
                    f.write(json.dumps(error["action"], indent=2, default=str))
                    f.write("\n```\n\n")
                
                # Include traceback if available
                if error.get("traceback"):
                    f.write("Traceback:\n```\n")
                    f.write(error["traceback"])
                    f.write("\n```\n\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Add automatic recommendations based on error patterns
            if stats['total_errors'] > 10:
                f.write("- Consider increasing retry delay and max retry attempts\n")
            
            if any("Timeout" in error_type for error_type in stats['error_types']):
                f.write("- Consider increasing page load timeout settings\n")
            
            if any("Navigation" in error_type for error_type in stats['error_types']):
                f.write("- Check for page navigation issues and network connectivity\n")
            
            if any("Element" in error_type for error_type in stats['error_types']):
                f.write("- Review element selection strategy and wait for elements to be visible\n")
        
        self.logger.info(f"Generated error report at {report_path}")
        return report_path
