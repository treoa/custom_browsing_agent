"""
Browser Monitoring Utilities

This module provides utilities for monitoring and tracking browser activity
for the Advanced Autonomous Research Agent system.
"""

import os
import time
import json
import asyncio
import tempfile
import logging
from typing import Dict, Any, Optional, List, Callable
from browser_use.browser.context import BrowserContext
from browser_use.agent.views import AgentHistoryList

logger = logging.getLogger(__name__)

class BrowserMonitor:
    """
    Monitor and log browser activity.
    
    This class provides methods for tracking browser actions, capturing
    screenshots, and recording browser state during research tasks.
    """
    
    def __init__(
        self,
        browser_context: BrowserContext,
        project_dir: str,
        logger=None,
        screenshot_interval: int = 5,
        enable_screenshots: bool = True
    ):
        """
        Initialize a BrowserMonitor instance.
        
        Args:
            browser_context: The browser context to monitor
            project_dir: Directory where the project is stored
            logger: Optional logger instance
            screenshot_interval: Interval in seconds between screenshots
            enable_screenshots: Whether to capture screenshots
        """
        self.browser_context = browser_context
        self.project_dir = project_dir
        self.logger = logger or logging.getLogger(__name__)
        self.screenshot_interval = screenshot_interval
        self.enable_screenshots = enable_screenshots
        
        # Create browser log directory
        self.browser_log_dir = os.path.join(project_dir, "logs", "browser")
        self.screenshots_dir = os.path.join(project_dir, "logs", "screenshots")
        os.makedirs(self.browser_log_dir, exist_ok=True)
        
        if enable_screenshots:
            os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Initialize browser log file
        self.browser_log_file = os.path.join(self.browser_log_dir, "browser_activity.log")
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.current_page_url = None
        self.page_history = []
        self.last_screenshot_time = 0
        self.last_screenshot_path = None
        self.screenshot_error_count = 0
        self.last_screenshot_error_time = 0
        self.screenshot_error_threshold = 3  # Max number of consecutive errors
        self.screenshot_error_cooldown = 60  # Time in seconds to wait after errors
        self.screenshots_enabled = enable_screenshots  # Track if screenshots are currently enabled
    
    async def start_monitoring(self):
        """Start monitoring browser activity."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_browser())
        self.logger.info("Browser monitoring started")
        
        # Log start of monitoring
        self._log_browser_event("monitoring_started", {
            "timestamp": time.time(),
            "message": "Browser monitoring started"
        })
    
    async def stop_monitoring(self):
        """Stop monitoring browser activity."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            try:
                self.monitoring_task.cancel()
                await asyncio.sleep(0.5)  # Give task time to cancel gracefully
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Browser monitoring stopped")
        
        # Log end of monitoring
        self._log_browser_event("monitoring_stopped", {
            "timestamp": time.time(),
            "message": "Browser monitoring stopped"
        })
    
    async def _monitor_browser(self):
        """Background task to monitor browser activity."""
        try:
            while self.is_monitoring:
                try:
                    # Get current browser state
                    state = await self.browser_context.get_state()
                    
                    # Check if URL has changed
                    if state.url != self.current_page_url:
                        self._log_url_change(self.current_page_url, state.url)
                        self.current_page_url = state.url
                        
                        # Add to page history
                        self.page_history.append({
                            "timestamp": time.time(),
                            "url": state.url,
                            "title": state.title
                        })
                        
                        # Trigger screenshot on page change if enabled
                        if self.screenshots_enabled and self.enable_screenshots:
                            await self._take_screenshot("page_change")
                    
                    # Periodic screenshots if enabled
                    if (self.screenshots_enabled and self.enable_screenshots and 
                            time.time() - self.last_screenshot_time >= self.screenshot_interval):
                        await self._take_screenshot("interval")
                    
                    # Check if we should re-enable screenshots after cooldown
                    if (not self.screenshots_enabled and 
                            time.time() - self.last_screenshot_error_time >= self.screenshot_error_cooldown):
                        self.logger.info("Re-enabling screenshots after cooldown period")
                        self.screenshots_enabled = True
                        self.screenshot_error_count = 0
                    
                except Exception as monitor_error:
                    self.logger.debug(f"Minor error in monitoring loop: {str(monitor_error)}")
                    # Don't break the loop for minor errors
                
                # Sleep for a short interval to prevent excessive CPU usage
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self.logger.debug("Browser monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in browser monitoring: {str(e)}")
    
    async def _take_screenshot(self, reason: str = "manual"):
        """
        Take a screenshot of the current browser state.
        
        Args:
            reason: Reason for taking the screenshot
        """
        # Skip if screenshots are disabled due to previous errors
        if not self.screenshots_enabled:
            return None
            
        try:
            session = await self.browser_context.get_session()
            if not session or not session.page:
                self.logger.debug("Cannot take screenshot: No active browser session")
                return None
            
            # Get timestamp for the filename
            timestamp = int(time.time())
            screenshot_path = os.path.join(self.screenshots_dir, f"screenshot_{timestamp}_{reason}.png")
            
            # Take screenshot
            await session.page.screenshot(path=screenshot_path)
            
            # Update state
            self.last_screenshot_time = time.time()
            self.last_screenshot_path = screenshot_path
            self.screenshot_error_count = 0  # Reset error count on success
            
            # Log screenshot event
            self._log_browser_event("screenshot_taken", {
                "timestamp": timestamp,
                "path": screenshot_path,
                "reason": reason,
                "url": self.current_page_url
            })
            
            self.logger.debug(f"Browser screenshot saved to {screenshot_path}")
            
            return screenshot_path
        except Exception as e:
            # Handle error quietly if it's just a screenshot
            self.screenshot_error_count += 1
            self.last_screenshot_error_time = time.time()
            
            # Log the error only at debug level to avoid spamming
            self.logger.debug(f"Error taking screenshot: {str(e)}")
            
            # If we've had too many consecutive errors, disable screenshots temporarily
            if self.screenshot_error_count >= self.screenshot_error_threshold:
                self.screenshots_enabled = False
                self.logger.info(
                    f"Temporarily disabling screenshots due to {self.screenshot_error_count} consecutive errors. "
                    f"Will retry in {self.screenshot_error_cooldown} seconds."
                )
                
            return None
    
    def _log_url_change(self, old_url: Optional[str], new_url: str):
        """
        Log a URL change event.
        
        Args:
            old_url: Previous URL
            new_url: New URL
        """
        self._log_browser_event("url_change", {
            "timestamp": time.time(),
            "old_url": old_url,
            "new_url": new_url
        })
        
        self.logger.info(f"Browser navigated from {old_url} to {new_url}")
    
    def _log_browser_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a browser event to the browser activity log.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        try:
            event = {
                "type": event_type,
                **data
            }
            
            with open(self.browser_log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.logger.error(f"Error logging browser event: {str(e)}")
    
    async def log_browser_action(self, action_type: str, action_data: Dict[str, Any]):
        """
        Log a browser action performed by the agent.
        
        Args:
            action_type: Type of action (e.g., "click", "type", "navigate")
            action_data: Data for the action
        """
        self._log_browser_event("agent_action", {
            "timestamp": time.time(),
            "action_type": action_type,
            "action_data": action_data
        })
        
        self.logger.info(f"Browser action: {action_type} - {json.dumps(action_data)}")
        
        # Take screenshot after action if enabled
        if self.screenshots_enabled and self.enable_screenshots:
            await self._take_screenshot("agent_action")
    
    def save_browser_history(self, path: Optional[str] = None) -> str:
        """
        Save browser history to a file.
        
        Args:
            path: Optional path to save history (defaults to browser_history.json in project dir)
            
        Returns:
            Path to the saved history file
        """
        history_path = path or os.path.join(self.project_dir, "logs", "browser_history.json")
        
        try:
            with open(history_path, "w") as f:
                json.dump(self.page_history, f, indent=2)
                
            self.logger.info(f"Browser history saved to {history_path}")
            return history_path
        except Exception as e:
            self.logger.error(f"Error saving browser history: {str(e)}")
            return ""
    
    async def record_agent_history(self, agent_history: AgentHistoryList, path: Optional[str] = None) -> str:
        """
        Save agent history to a file and take screenshots of key steps.
        
        Args:
            agent_history: Agent history list to record
            path: Optional path to save history (defaults to agent_history.json in project dir)
            
        Returns:
            Path to the saved history file
        """
        history_path = path or os.path.join(self.project_dir, "logs", "agent_history.json")
        
        try:
            # Save agent history as JSON
            agent_dict = agent_history.model_dump()
            with open(history_path, "w") as f:
                json.dump(agent_dict, f, indent=2)
            
            self.logger.info(f"Agent history saved to {history_path}")
            
            # Also save as a readable markdown file for human review
            md_path = os.path.join(self.project_dir, "logs", "agent_history.md")
            
            with open(md_path, "w") as f:
                f.write("# Agent Activity History\n\n")
                
                for i, step in enumerate(agent_history.history):
                    f.write(f"## Step {i+1}\n\n")
                    
                    # Write step details
                    f.write(f"**URL:** {step.state.url if hasattr(step.state, 'url') else 'N/A'}\n\n")
                    f.write(f"**Title:** {step.state.title if hasattr(step.state, 'title') else 'N/A'}\n\n")
                    
                    # Write actions
                    f.write("### Actions\n\n")
                    for j, action in enumerate(step.actions):
                        f.write(f"{j+1}. {str(action)}\n")
                    
                    f.write("\n")
                    
                    # Write results
                    f.write("### Results\n\n")
                    for j, result in enumerate(step.result):
                        if result.error:
                            f.write(f"{j+1}. ❌ **Error:** {result.error}\n\n")
                        elif result.extracted_content:
                            f.write(f"{j+1}. ✅ **Extracted Content:**\n\n")
                            f.write(f"```\n{result.extracted_content[:500]}{'...' if len(result.extracted_content) > 500 else ''}\n```\n\n")
                        elif result.is_done:
                            f.write(f"{j+1}. ✅ **Task Complete**\n\n")
                        else:
                            f.write(f"{j+1}. ✅ **Success**\n\n")
                    
                    f.write("---\n\n")
            
            self.logger.info(f"Agent history markdown saved to {md_path}")
            
            return history_path
        except Exception as e:
            self.logger.error(f"Error recording agent history: {str(e)}")
            return ""
