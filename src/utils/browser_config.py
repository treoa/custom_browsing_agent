"""
Browser Configuration Utilities

This module provides utilities for configuring and connecting to browsers
for the Advanced Autonomous Research Agent system.
"""

import os
import subprocess
import time
import socket
import platform
import json
import logging
from typing import Optional, Dict, Any, Tuple
import urllib.request
import urllib.error

from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from browser_use.utils import logger

class BrowserConfigUtils:
    """
    Utilities for browser configuration and connection.
    
    This class provides methods for configuring browsers and connecting to
    existing browser instances, with enhanced features for the Advanced
    Autonomous Research Agent system.
    """
    
    @staticmethod
    def create_default_browser(headless: bool = False, timeout: int = 30, viewport: Dict[str, int] = None) -> Browser:
        """
        Create a default browser configuration with enhanced options.
        
        Args:
            headless: Whether to run the browser in headless mode (default: False)
            timeout: Maximum time to wait for page loads in seconds (default: 30)
            viewport: Optional viewport size override as {width, height}
            
        Returns:
            Configured Browser instance
        """
        # Set reasonable default viewport if not provided
        if viewport is None:
            viewport = {"width": 1280, "height": 800}
        
        # Use only the parameters that are actually supported in browser-use
        browser_config = BrowserConfig(
            headless=headless,
            new_context_config=BrowserContextConfig(
                minimum_wait_page_load_time=1,
                wait_for_network_idle_page_load_time=2.0,
                maximum_wait_page_load_time=timeout,
                browser_window_size=BrowserContextWindowSize(
                    width=viewport.get("width", 1280),
                    height=viewport.get("height", 800)
                ),
                highlight_elements=True,
                viewport_expansion=1000,  # Capture more of the page for better context
            )
        )
        
        return Browser(config=browser_config)
    
    @staticmethod
    def create_chrome_browser(headless: bool = False, user_data_dir: Optional[str] = None, debugging_port: int = 9222, debugging_host: str = "localhost") -> Browser:
        """
        Create a Chrome browser instance with optimal configuration for research.
        
        Args:
            headless: Whether to run in headless mode (default: False)
            user_data_dir: Optional path to Chrome user data directory
            debugging_port: Chrome debugging port (default: 9222)
            debugging_host: Chrome debugging host (default: localhost)
            
        Returns:
            Configured Browser instance optimized for research
        """
        # Check if Chrome is running in debugging mode first
        logger.info(f"Checking for Chrome running in debugging mode on {debugging_host}:{debugging_port}")
        chrome_running, websocket_url = BrowserConfigUtils._is_chrome_debugging_running(port=debugging_port, host=debugging_host)
        
        # Configure viewport for optimal content extraction
        research_viewport = BrowserContextWindowSize(width=1600, height=900)
        
        # Default temp user data directory if none provided
        if not user_data_dir:
            temp_dir = os.path.join(os.path.expanduser("~"), ".browser-use-profiles", "research-profile")
            os.makedirs(temp_dir, exist_ok=True)
            user_data_dir = temp_dir
            logger.info(f"Using temporary Chrome profile at: {temp_dir}")
        
        # Find Chrome executable path
        chrome_path = BrowserConfigUtils.get_default_chrome_path()
        if not chrome_path:
            logger.warning("Could not find Chrome executable, will use default browser")
            
        if chrome_running and websocket_url:
            logger.info(f"Connecting to existing Chrome instance using {websocket_url}")
            # Try to connect to existing Chrome using WebSocket URL
            try:
                # Using cdp_url parameter with websocket URL
                browser_config = BrowserConfig(
                    headless=False,  # Connect to visible browser
                    cdp_url=websocket_url,
                    new_context_config=BrowserContextConfig(
                        minimum_wait_page_load_time=1,
                        wait_for_network_idle_page_load_time=3.0,
                        maximum_wait_page_load_time=30,
                        browser_window_size=research_viewport,
                        highlight_elements=True,
                    )
                )
                return Browser(config=browser_config)
            except Exception as e:
                logger.error(f"Failed to connect to existing Chrome via WebSocket URL: {str(e)}")
                logger.info("Trying alternative connection method...")
                
                # Try alternative connection method using HTTP endpoint
                try:
                    browser_config = BrowserConfig(
                        headless=False,  # Connect to visible browser
                        cdp_url=f"http://{debugging_host}:{debugging_port}",
                        new_context_config=BrowserContextConfig(
                            minimum_wait_page_load_time=1,
                            wait_for_network_idle_page_load_time=3.0,
                            maximum_wait_page_load_time=30,
                            browser_window_size=research_viewport,
                            highlight_elements=True,
                        )
                    )
                    return Browser(config=browser_config)
                except Exception as e2:
                    logger.error(f"Failed with alternative connection method: {str(e2)}")
                    logger.info("Falling back to launching a new browser")
        
        # If we can't connect to an existing Chrome or none is running,
        # try to start a new Chrome instance
        if chrome_path and not chrome_running:
            logger.info(f"Starting new Chrome instance using: {chrome_path}")
            BrowserConfigUtils._start_chrome_with_debugging(
                chrome_path=chrome_path,
                user_data_dir=user_data_dir,
                debugging_port=debugging_port,
                debugging_host=debugging_host
            )
            # Wait for Chrome to start
            time.sleep(5)
            
            # Check if Chrome is running now
            chrome_running, websocket_url = BrowserConfigUtils._is_chrome_debugging_running(port=debugging_port, host=debugging_host)
            
            # Try to connect to the Chrome we just started
            if chrome_running and websocket_url:
                try:
                    logger.info(f"Connecting to newly started Chrome using {websocket_url}")
                    browser_config = BrowserConfig(
                        headless=False,  # Connect to visible browser
                        cdp_url=websocket_url,
                        new_context_config=BrowserContextConfig(
                            minimum_wait_page_load_time=1,
                            wait_for_network_idle_page_load_time=3.0,
                            maximum_wait_page_load_time=30,
                            browser_window_size=research_viewport,
                            highlight_elements=True,
                        )
                    )
                    return Browser(config=browser_config)
                except Exception as e:
                    logger.error(f"Failed to connect to Chrome we launched: {str(e)}")
                    logger.info("Falling back to default browser")
        
        # If everything else fails, create a new browser instance using default launcher
        logger.info("Creating default browser instance")
        browser_config = BrowserConfig(
            headless=headless,
            # Configure browser launch options - removed problematic flags
            launch_options={
                # Disable unnecessary features to improve performance
                "args": [
                    "--disable-extensions",
                    "--disable-notifications",
                    "--disable-popup-blocking",
                    "--disable-infobars",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                # Set user data directory if provided
                "userDataDir": user_data_dir
            },
            new_context_config=BrowserContextConfig(
                minimum_wait_page_load_time=1,
                wait_for_network_idle_page_load_time=3.0,
                maximum_wait_page_load_time=30,
                browser_window_size=research_viewport,
                highlight_elements=True,
            )
        )
        
        return Browser(config=browser_config)
    
    @staticmethod
    def create_browser_with_existing_chrome(
        chrome_path: str,
        user_data_dir: str,
        debugging_port: int = 9222,
        debugging_host: str = "localhost",
        start_if_not_running: bool = True
    ) -> Browser:
        """
        Create a browser configuration that connects to an existing Chrome instance.
        
        Args:
            chrome_path: Path to Chrome executable
            user_data_dir: Path to Chrome user data directory
            debugging_port: Chrome debugging port (default: 9222)
            debugging_host: Chrome debugging host (default: localhost)
            start_if_not_running: Whether to start Chrome if not already running (default: True)
            
        Returns:
            Configured Browser instance
        """
        # Check if Chrome is already running with remote debugging enabled
        chrome_running, _ = BrowserConfigUtils._is_chrome_debugging_running(debugging_port, debugging_host)
        
        # Start Chrome with remote debugging if not running and requested
        if not chrome_running and start_if_not_running:
            BrowserConfigUtils._start_chrome_with_debugging(
                chrome_path=chrome_path,
                user_data_dir=user_data_dir,
                debugging_port=debugging_port,
                debugging_host=debugging_host
            )
            # Wait for Chrome to start
            time.sleep(5)
        
        # Create a browser instance that connects to Chrome
        browser = BrowserConfigUtils.create_chrome_browser(
            headless=False, 
            user_data_dir=user_data_dir,
            debugging_port=debugging_port,
            debugging_host=debugging_host
        )
        return browser
    
    @staticmethod
    def get_default_chrome_path() -> Optional[str]:
        """
        Get the default Chrome path based on the operating system.
        
        Returns:
            Path to Chrome executable or None if not found
        """
        import platform
        import os
        
        system = platform.system()
        
        if system == "Windows":
            # Check common Windows Chrome locations
            locations = [
                os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe")
            ]
            
            for location in locations:
                if os.path.exists(location):
                    return location
        
        elif system == "Darwin":  # macOS
            locations = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
            ]
            
            for location in locations:
                if os.path.exists(location):
                    return location
        
        elif system == "Linux":
            # Check common Linux Chrome/Chromium locations
            locations = [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
                "/snap/bin/chromium"
            ]
            
            for location in locations:
                if os.path.exists(location):
                    return location
        
        return None
    
    @staticmethod
    def _is_chrome_debugging_running(port: int = 9222, host: str = "localhost") -> Tuple[bool, Optional[str]]:
        """
        Check if Chrome is running with remote debugging enabled and get the WebSocket URL.
        
        Args:
            port: Chrome debugging port (default: 9222)
            host: Chrome debugging host (default: localhost)
            
        Returns:
            Tuple of (is_running, websocket_url)
        """
        websocket_url = None
        try:
            # First try to connect to the debugging port
            with socket.create_connection((host, port), timeout=1):
                # Then try to fetch the JSON endpoint to verify it's responding correctly
                try:
                    # First try the /json/version endpoint
                    url = f"http://{host}:{port}/json/version"
                    with urllib.request.urlopen(url, timeout=2) as response:
                        data = json.loads(response.read().decode())
                        if 'webSocketDebuggerUrl' in data:
                            websocket_url = data['webSocketDebuggerUrl']
                            logger.info(f"Found Chrome debugging WebSocket URL: {websocket_url}")
                            return True, websocket_url
                except (urllib.error.URLError, json.JSONDecodeError) as e1:
                    # If /json/version fails, try /json endpoint
                    try:
                        url = f"http://{host}:{port}/json"
                        with urllib.request.urlopen(url, timeout=2) as response:
                            data = json.loads(response.read().decode())
                            if isinstance(data, list) and len(data) > 0 and 'webSocketDebuggerUrl' in data[0]:
                                websocket_url = data[0]['webSocketDebuggerUrl']
                                logger.info(f"Found Chrome debugging WebSocket URL from /json: {websocket_url}")
                                return True, websocket_url
                    except (urllib.error.URLError, json.JSONDecodeError) as e2:
                        logger.warning(f"Port {port} is open but Chrome debugging API is not responding properly: {e1}, {e2}")
                        websocket_url = f"ws://{host}:{port}/devtools/browser"
                        logger.info(f"Using fallback WebSocket URL: {websocket_url}")
                        return True, websocket_url
                
                # If we reach here, we're connected but couldn't get a WebSocket URL
                logger.warning(f"Connected to {host}:{port} but couldn't get WebSocket URL")
                websocket_url = f"ws://{host}:{port}/devtools/browser"
                logger.info(f"Using fallback WebSocket URL: {websocket_url}")
                return True, websocket_url
        except (socket.timeout, ConnectionRefusedError):
            return False, None
        
        return False, None
    
    @staticmethod
    def _start_chrome_with_debugging(
        chrome_path: str,
        user_data_dir: str,
        debugging_port: int = 9222,
        debugging_host: str = "localhost"
    ) -> None:
        """
        Start Chrome with remote debugging enabled.
        
        Args:
            chrome_path: Path to Chrome executable
            user_data_dir: Path to Chrome user data directory
            debugging_port: Chrome debugging port (default: 9222)
            debugging_host: Chrome debugging host (default: localhost)
        """
        # Ensure user_data_dir exists
        user_data_dir = os.path.expanduser(user_data_dir)
        os.makedirs(user_data_dir, exist_ok=True)
        
        # Prepare Chrome command line arguments - removed problematic flags
        chrome_args = [
            chrome_path,
            f"--remote-debugging-port={debugging_port}",
            f"--remote-debugging-address={debugging_host}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--start-maximized",
            # Add research-friendly settings
            "--disable-extensions",
            "--disable-notifications",
            "--disable-popup-blocking",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            # Removed problematic flags like --disable-web-security
        ]
        
        logger.info(f"Starting Chrome with command: {' '.join(chrome_args)}")
        
        # Start Chrome as a background process
        if platform.system() == "Windows":
            subprocess.Popen(
                chrome_args,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                chrome_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
