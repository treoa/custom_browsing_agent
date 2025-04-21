#!/usr/bin/env python3
"""
Advanced Autonomous Research Agent

This system implements a multi-agent architecture for conducting deep, methodical 
research using browser automation. It coordinates specialized agents for different
aspects of the research process.
"""

import os
import asyncio
import time
import json
import argparse
import logging
import uuid
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import socket
import platform
import urllib.request
from rich.console import Console
from rich.panel import Panel
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='research_agent.log'
)
logger = logging.getLogger('research_agent')

# Rich console for pretty terminal output
console = Console()

# Load environment variables from .env file
load_dotenv()

# Import after loading environment variables to ensure API keys are available
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
    from langchain_fireworks import ChatFireworks
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import SystemMessage, HumanMessage
    from browser_use import Browser, BrowserConfig
    from browser_use.browser.context import BrowserContextConfig
    from browser_use.agent.service import Agent as BrowserAgent
except ImportError as e:
    console.print(Panel(f"[bold red]Import Error:[/] {str(e)}\nMake sure all required packages are installed.", border_style="red"))
    logger.error(f"Import error: {str(e)}")
    sys.exit(1)

# Check and configure browser-use version
def check_browser_use_version():
    """Verify the installed browser-use version."""
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("browser-use").version
        if version != "0.1.14":
            console.print(Panel(f"[bold yellow]Warning:[/] Using browser-use version {version}. This project is designed for version 0.1.14.", border_style="yellow"))
    except Exception as e:
        console.print(Panel(f"[bold yellow]Warning:[/] Could not verify browser-use version: {e}", border_style="yellow"))

# Browser setup functions
def is_chrome_debugging_running(port=9222, host="localhost"):
    """Check if Chrome is running with remote debugging enabled."""
    try:
        with socket.create_connection((host, port), timeout=1):
            try:
                url = f"http://{host}:{port}/json/version"
                with urllib.request.urlopen(url, timeout=2) as response:
                    data = json.loads(response.read().decode())
                    if 'webSocketDebuggerUrl' in data:
                        return True, data['webSocketDebuggerUrl']
            except Exception:
                return False, None
    except Exception:
        return False, None
    return False, None

def get_default_chrome_path():
    """Get the default Chrome path based on operating system."""
    system = platform.system()
    
    chrome_path = os.getenv("CHROME_PATH", "")
    if chrome_path and os.path.exists(chrome_path):
        return chrome_path
        
    if system == "Windows":
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

def start_chrome_with_debugging(port=9222, host="localhost"):
    """Start Chrome with remote debugging enabled if not already running."""
    # Check if Chrome is already running with debugging
    chrome_running, websocket_url = is_chrome_debugging_running(port=port, host=host)
    
    if chrome_running:
        console.print(f"[green]Chrome is already running with debugging on {host}:{port}[/]")
        return True, websocket_url

    # Get Chrome path
    chrome_path = get_default_chrome_path()
    if not chrome_path:
        console.print("[bold red]Error: Could not find Chrome executable.[/]")
        return False, None
        
    # Set up user data directory
    user_data_dir = os.getenv("CHROME_USER_DATA", 
                            os.path.join(os.path.expanduser("~"), ".browser-use-profiles", "research-profile"))
    os.makedirs(user_data_dir, exist_ok=True)
    
    # Chrome command line arguments
    chrome_args = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--remote-debugging-address={host}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
        "--disable-extensions",
        "--disable-notifications",
        "--disable-popup-blocking",
    ]
    
    console.print(f"[blue]Starting Chrome with debugging on {host}:{port}...[/]")
    
    try:
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
        
        # Wait for Chrome to start
        for _ in range(10):
            time.sleep(1)
            chrome_running, websocket_url = is_chrome_debugging_running(port=port, host=host)
            if chrome_running:
                console.print("[green]Chrome started successfully with debugging enabled[/]")
                return True, websocket_url
                
        console.print("[bold yellow]Warning: Chrome might not have started properly with debugging.[/]")
        return False, None
    except Exception as e:
        console.print(f"[bold red]Error starting Chrome: {str(e)}[/]")
        return False, None

# Custom OpenRouter integration for LLM access
class ChatOpenRouter(ChatOpenAI):
    """
    Custom implementation of ChatOpenAI that uses OpenRouter's API endpoint.
    OpenRouter is API-compatible with OpenAI but routes to many different models.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        openrouter_api_key: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        # Get API key from args or environment
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Please provide it as an argument or set the OPENROUTER_API_KEY environment variable."
            )
            
        # Initialize with OpenRouter API base
        super().__init__(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            **kwargs
        )

def get_available_models():
    """
    Get available models based on API keys in the environment.
    
    Returns:
        Dict containing available models by provider
    """
    available = {
        "openai": [],
        "gemini": [],
        "anthropic": [],
        "openrouter": [],
        "fireworks": []
    }
    
    # Check OpenAI models
    if os.getenv("OPENAI_API_KEY"):
        available["openai"] = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]
    
    # Check Google models
    if os.getenv("GEMINI_API_KEY"):
        available["gemini"] = ["gemini-2.0-flash", "gemini-2.5-pro-exp-03-25", "gemini-2.0-flash-lite"]
    
    # Check Anthropic models
    if os.getenv("ANTHROPIC_API_KEY"):
        available["anthropic"] = ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"]
    
    # Check OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        available["openrouter"] = ["*"]  # Can route to many models
    
    # Check Fireworks
    if os.getenv("FIREWORKS_API_KEY"):
        available["fireworks"] = ["fireworks-mixtral", "fireworks-llama"]
    
    return available

def initialize_llm(model_name, model_type="basic"):
    """
    Initialize a language model with proper error handling.
    
    Args:
        model_name: The name of the model to initialize
        model_type: Whether this is a "basic" or "advanced" model (for logging)
        
    Returns:
        Tuple of (model instance, error message)
    """
    if not model_name:
        return None, f"No {model_type} model specified and no default could be determined from available API keys"
    
    try:
        # Check OpenAI models
        if model_name.startswith("gpt-"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None, f"OpenAI API key (OPENAI_API_KEY) is required for model: {model_name}"
            return ChatOpenAI(model=model_name, temperature=0.1), None
        
        # Check Google models
        elif model_name.startswith("gemini-"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None, f"Google Gemini API key (GEMINI_API_KEY) is required for model: {model_name}"
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.1, api_key=api_key), None
        
        # Check Anthropic models
        elif model_name.startswith("claude-"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None, f"Anthropic API key (ANTHROPIC_API_KEY) is required for model: {model_name}"
            return ChatAnthropic(model=model_name, temperature=0.1, api_key=api_key), None
        
        # Check Fireworks models
        elif "fireworks" in model_name:
            api_key = os.getenv("FIREWORKS_API_KEY")
            if not api_key:
                return None, f"Fireworks API key (FIREWORKS_API_KEY) is required for model: {model_name}"
            return ChatFireworks(model=model_name, temperature=0.1, api_key=api_key), None
        # Use OpenRouter for any other model
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return None, f"OpenRouter API key (OPENROUTER_API_KEY) is required for model: {model_name}"
            return ChatOpenRouter(model=model_name, temperature=0.1, openrouter_api_key=api_key), None
    
    except Exception as e:
        return None, f"Error initializing {model_type} model ({model_name}): {str(e)}"

async def run_research(
    query: str,
    basic_model: str,
    advanced_model: str,
    max_steps: int = 50,
    storage_path: str = "./research_data",
    browser_visible: bool = True
):
    """
    Execute a research task with the multi-agent system using browser automation.
    
    Args:
        query: The research query to investigate
        basic_model: LLM model to use for basic tasks
        advanced_model: LLM model to use for complex tasks
        max_steps: Maximum number of research steps
        storage_path: Path for storing research data
        browser_visible: Whether to show the browser during research
        
    Returns:
        Research results
    """
    try:
        start_time = time.time()
        
        # Check if we have any models specified
        if not basic_model and not advanced_model:
            # Check for available API keys and suggest models
            available_models = get_available_models()
            api_suggestions = []
            
            if not available_models["openai"] and not available_models["gemini"] and not available_models["anthropic"]:
                api_suggestions = [
                    "OPENAI_API_KEY for OpenAI models (gpt-3.5-turbo, gpt-4)",
                    "GEMINI_API_KEY for Google models (gemini-2.0-flash, gemini-2.5-pro)",
                    "ANTHROPIC_API_KEY for Claude models (claude-3-haiku, claude-3-opus)",
                    "OPENROUTER_API_KEY for access to many models through OpenRouter"
                ]
                
                error_msg = (
                    "No language model credentials found. Please set at least one of these API keys in your .env file:\n" +
                    "\n".join(f"- {suggestion}" for suggestion in api_suggestions)
                )
                console.print(Panel(f"[bold red]Error:[/] {error_msg}", border_style="red"))
                return f"Error: {error_msg}"
                
        # Check browser-use version
        check_browser_use_version()

        # Auto-start Chrome browser if needed
        debugging_port = int(os.getenv("CHROME_DEBUGGING_PORT", "9222"))
        debugging_host = os.getenv("CHROME_DEBUGGING_HOST", "localhost")
        chrome_started, websocket_url = start_chrome_with_debugging(port=debugging_port, host=debugging_host)
        
        if not chrome_started:
            error_msg = f"Could not start or connect to Chrome browser. Please start Chrome manually with debugging enabled on port {debugging_port}."
            console.print(Panel(f"[bold red]Error:[/] {error_msg}", border_style="red"))
            logger.error(error_msg)
            return f"Error: {error_msg}"

        # Create a unique ID for this research session
        research_id = str(uuid.uuid4())
        
        # Create directories for research data
        session_path = os.path.join(storage_path, f"research_{research_id}")
        os.makedirs(session_path, exist_ok=True)
        
        log_dir = os.path.join(session_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        screenshots_dir = os.path.join(log_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        memory_dir = os.path.join(session_path, "memory")
        os.makedirs(memory_dir, exist_ok=True)
        
        results_dir = os.path.join(session_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        sources_dir = os.path.join(session_path, "sources")
        os.makedirs(sources_dir, exist_ok=True)
        
        # Set up file handler for logging
        log_file = os.path.join(log_dir, "research.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log start of research
        logger.info(f"Starting research on query: {query}")
        logger.info(f"Using basic model: {basic_model}")
        logger.info(f"Using advanced model: {advanced_model}")
        logger.info(f"Session ID: {research_id}")
        
        # Initialize status file
        status_file = os.path.join(session_path, "status.md")
        with open(status_file, "w") as f:
            f.write(f"# Research Status\n\n")
            f.write(f"Query: {query}\n\n")
            f.write(f"Status: Initializing\n\n")
            f.write(f"Progress: 0%\n\n")
            f.write(f"Current task: Setting up research environment\n\n")
        
        # Initialize todo file
        todo_file = os.path.join(session_path, "todo.md")
        with open(todo_file, "w") as f:
            f.write(f"# Research Tasks\n\n")
            f.write(f"## Query\n\n{query}\n\n")
            f.write(f"## Pending Tasks\n\n- Initial research plan creation\n")
        
        # Create project info file
        project_file = os.path.join(session_path, "project_info.md")
        with open(project_file, "w") as f:
            f.write(f"# Research Project: {query}\n\n")
            f.write(f"Session ID: {research_id}\n\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Configuration\n\n")
            f.write(f"- Basic Model: {basic_model}\n")
            f.write(f"- Advanced Model: {advanced_model}\n")
            f.write(f"- Max Steps: {max_steps}\n")
            f.write(f"- Browser Visible: {browser_visible}\n\n")
        
        # Update status function
        def update_progress(step, total, description):
            progress = int((step / total) * 100)
            logger.info(f"Progress: {progress}% - {description}")
            
            with open(status_file, "w") as f:
                f.write(f"# Research Status\n\n")
                f.write(f"Query: {query}\n\n")
                f.write(f"Status: In Progress\n\n")
                f.write(f"Progress: {progress}%\n\n")
                f.write(f"Current task: {description}\n\n")
                f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if progress_callback:
                progress_callback(f"Step {step}/{total} ({progress}%): {description}")
        
        # Initialize progress callback
        progress_callback = None
        
        # Initialize language models
        update_progress(1, max_steps, "Initializing language models")
        
        # Set up the basic LLM using the new initialization function
        basic_llm, basic_error = initialize_llm(basic_model, "basic")
        if basic_error:
            logger.error(basic_error)
            console.print(Panel(f"[bold red]Error:[/] {basic_error}", border_style="red"))
        
        # Set up the advanced LLM
        advanced_llm, advanced_error = initialize_llm(advanced_model, "advanced")
        if advanced_error:
            logger.error(advanced_error)
            console.print(Panel(f"[bold red]Error:[/] {advanced_error}", border_style="red"))
        
        # Check if at least one LLM was initialized
        if not basic_llm and not advanced_llm:
            error_msg = "Failed to initialize any language models. Please check your API keys and internet connection."
            logger.error(error_msg)
            console.print(Panel(f"[bold red]Error:[/] {error_msg}", border_style="red"))
            return f"Error: {error_msg}"
            
        # Use at least one available model
        if not basic_llm:
            basic_llm = advanced_llm
            logger.info(f"Using advanced model for basic tasks due to initialization failure")
        if not advanced_llm:
            advanced_llm = basic_llm
            logger.info(f"Using basic model for advanced tasks due to initialization failure")
        
        # Initialize browser and browser agent
        update_progress(2, max_steps, "Setting up browser environment")

        # Attempt to initialize browser and set up a research agent
        try:
            logger.info("Initializing browser with CDP URL: %s", websocket_url or f"http://{debugging_host}:{debugging_port}")
            
            # Configure browser instance
            browser_config = BrowserConfig(
                headless=not browser_visible,
                cdp_url=websocket_url or f"http://{debugging_host}:{debugging_port}",
                new_context_config=BrowserContextConfig(
                    minimum_wait_page_load_time=1,
                    wait_for_network_idle_page_load_time=3.0,
                    maximum_wait_page_load_time=30,
                    highlight_elements=True,
                )
            )
            
            browser = Browser(config=browser_config)
            logger.info("Browser initialized successfully")
            
            # Create a new browser context
            browser_context = await browser.new_context()
            logger.info("Browser context created successfully")
            
            # Initialize browser monitor for logging
            browser_monitor = None
            
            # Currently not importing ResearchAgent due to circular import
            # Will be implemented directly in run_research flow
            from src.research_agent import ResearchAgent
            
            # Initialize research agent
            research_agent = ResearchAgent(
                basic_llm=basic_llm,
                advanced_llm=advanced_llm,
                browser=browser,
                browser_context=browser_context,
                max_steps=max_steps,
                storage_path=session_path,
                logger=logger,
                progress_callback=update_progress,
            )
            
            # Run the research
            update_progress(3, max_steps, "Starting research execution")
            results = await research_agent.research(
                query=query,
                context={"storage_path": session_path}
            )
            
            # Save results
            output_file = os.path.join(results_dir, "research_output.md")
            with open(output_file, "w") as f:
                f.write(f"# Research Results: {query}\n\n")
                f.write(f"## Executive Summary\n\n")
                f.write(f"{results.get('summary', 'No summary available.')}\n\n")
                f.write(f"## Detailed Findings\n\n")
                f.write(f"{results.get('findings', 'No detailed findings available.')}\n\n")
                f.write(f"## Sources\n\n")
                for source in results.get("sources", []):
                    f.write(f"- [{source.get('title', 'Unnamed Source')}]({source.get('url', '')})\n")
            
            # Also save as main project output
            with open(os.path.join(session_path, "research_output.md"), "w") as f:
                f.write(f"# Research Results: {query}\n\n")
                f.write(f"## Executive Summary\n\n")
                f.write(f"{results.get('summary', 'No summary available.')}\n\n")
                f.write(f"## Detailed Findings\n\n")
                f.write(f"{results.get('findings', 'No detailed findings available.')}\n\n")
                f.write(f"## Sources\n\n")
                for source in results.get("sources", []):
                    f.write(f"- [{source.get('title', 'Unnamed Source')}]({source.get('url', '')})\n")
            
            # Update status to complete
            with open(status_file, "w") as f:
                f.write(f"# Research Status\n\n")
                f.write(f"Query: {query}\n\n")
                f.write(f"Status: Completed\n\n")
                f.write(f"Progress: 100%\n\n")
                f.write(f"Duration: {time.time() - start_time:.2f} seconds\n\n")
                f.write(f"Results: [research_output.md](research_output.md)\n\n")
                f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Update todo file with completed tasks
            update_todo_file(todo_file, results)
            
            console.print(Panel(f"[bold green]Research completed successfully![/]\n\nResults saved to: {output_file}", border_style="green"))
            
            return str(results)
        except Exception as e:
            error_msg = f"Error initializing browser or running research: {str(e)}"
            logger.error(error_msg, exc_info=True)
            console.print(Panel(f"[bold red]Error:[/] {error_msg}", border_style="red"))
            return f"Error: {error_msg}"
    
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/] {str(e)}", border_style="red"))
        logger.error(f"Research error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"
    finally:
        # Ensure browser is properly closed
        try:
            if browser_monitor:
                try:
                    await browser_monitor.stop_monitoring()
                    logger.info("Browser monitoring stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping browser monitoring: {str(e)}")
            
            # Close browser context
            try:
                if browser_context:
                    await browser_context.close()
                    logger.info("Browser context closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser context: {str(e)}")
            
            # Close browser
            try:
                if browser:
                    await browser.close()
                    logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
                
            logger.info("Browser resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up browser resources: {str(e)}")

def update_todo_file(todo_path: str, results: dict) -> None:
    """Update the todo.md file with completed tasks."""
    if os.path.exists(todo_path):
        # Read existing content
        with open(todo_path, "r") as f:
            content = f.read()
        
        # Add completed tasks
        completed_tasks = results.get("execution_state", {}).get("completed_tasks", {})
        
        new_content = content + "\n\n## Completed Tasks\n\n"
        for task_id in completed_tasks:
            new_content += f"- [x] {task_id}\n"
        
        # Add pending tasks
        pending_tasks = results.get("execution_state", {}).get("pending_tasks", [])
        if pending_tasks:
            new_content += "\n\n## Pending Tasks\n\n"
            for task_id in pending_tasks:
                new_content += f"- [ ] {task_id}\n"
        
        # Write updated content
        with open(todo_path, "w") as f:
            f.write(new_content)

def create_ui():
    """
    Create the Gradio UI for the Research Agent.
    
    Returns:
        Gradio interface
    """
    # Get available models based on API keys
    available_models = get_available_models()
    
    # Create combined list of models
    model_choices = []
    model_provider_map = {}
    
    # Add OpenAI models
    for model in available_models["openai"]:
        model_choices.append(model)
        model_provider_map[model] = "OpenAI"
    
    # Add Google models
    for model in available_models["gemini"]:
        model_choices.append(model)
        model_provider_map[model] = "Google" 
    
    # Add OpenRouter indication if available
    if available_models["openrouter"]:
        model_choices.append("Model through OpenRouter")
        model_provider_map["Model through OpenRouter"] = "OpenRouter"
    
    # Add Fireworks models
    for model in available_models["fireworks"]:
        model_choices.append(model)
        model_provider_map[model] = "Fireworks"
    
    # Set default models or add guidance
    if not model_choices:
        basic_model_default = "No models available - add API keys"
        advanced_model_default = "No models available - add API keys"
        model_choices = ["No models available - add API keys"]
    else:
        basic_model_default = model_choices[0]
        # Use the most capable model for advanced if available
        if len(model_choices) > 1:
            for preferred in ["gpt-4-turbo", "gpt-4o", "claude-3-opus", "gemini-2.5-pro-exp-03-25"]:
                if preferred in model_choices:
                    advanced_model_default = preferred
                    break
            else:
                advanced_model_default = model_choices[0]
        else:
            advanced_model_default = model_choices[0]
    
    with gr.Blocks(title='Advanced Autonomous Research Agent') as interface:
        gr.Markdown('# Advanced Autonomous Research Agent')
        gr.Markdown('This system uses a multi-agent architecture for conducting deep, methodical research.')
        
        # Add API key guidance if no models available
        if not available_models["openai"] and not available_models["gemini"] and not available_models["anthropic"]:
            gr.Markdown("""
            ### ⚠️ No API Keys Detected
            
            You need to set at least one LLM API key in your .env file:
            
            - `OPENAI_API_KEY` for GPT models
            - `GEMINI_API_KEY` for Google models
            - `ANTHROPIC_API_KEY` for Claude models
            - `OPENROUTER_API_KEY` for routing to various models
            """)
        
        with gr.Row():
            with gr.Column():
                # Input components
                query = gr.Textbox(
                    label='Research Query',
                    placeholder='E.g., What are the latest advancements in quantum computing?',
                    lines=3,
                )
                with gr.Row():
                    basic_model = gr.Dropdown(
                        choices=model_choices, 
                        label='Basic Model (for routine tasks)', 
                        value=basic_model_default,
                        info="Select a model for basic research tasks"
                    )
                    advanced_model = gr.Dropdown(
                        choices=model_choices, 
                        label='Advanced Model (for complex tasks)', 
                        value=advanced_model_default,
                        info="Select a model for complex reasoning and synthesis"
                    )
                max_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Maximum Research Steps"
                )
                storage_path = gr.Textbox(
                    label='Storage Path',
                    value='./research_data',
                )
                browser_visible = gr.Checkbox(
                    label='Show Browser UI',
                    value=True,
                    info="Display the browser while research is conducted"
                )
                submit_btn = gr.Button('Start Research', variant="primary")
            
            with gr.Column():
                # Output components
                output = gr.Markdown(label='Research Results')
                progress = gr.Textbox(label='Progress', lines=10, interactive=False)
        
        # Set up event handlers
        submit_btn.click(
            fn=lambda *args: asyncio.run(run_research(*args)),
            inputs=[query, basic_model, advanced_model, max_steps, storage_path, browser_visible],
            outputs=output,
        )
    
    return interface

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    # Check available API keys to set reasonable defaults
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    
    # Set default models based on available credentials
    default_basic = "gpt-3.5-turbo" if has_openai else "gemini-2.0-flash" if has_gemini else "claude-3-haiku" if has_anthropic else None
    default_advanced = "gpt-4-turbo" if has_openai else "gemini-2.5-pro-exp-03-25" if has_gemini else "claude-3-opus" if has_anthropic else None
    
    parser = argparse.ArgumentParser(description='Advanced Autonomous Research Agent')
    parser.add_argument('--query', type=str, help='Research query to investigate')
    parser.add_argument('--basic-model', type=str, default=default_basic, 
                       help='LLM model to use for basic tasks')
    parser.add_argument('--advanced-model', type=str, default=default_advanced, 
                       help='LLM model to use for complex tasks')
    parser.add_argument('--max-steps', type=int, default=50,
                       help='Maximum number of research steps (default: 50)')
    parser.add_argument('--storage-path', type=str, default='./research_data',
                       help='Path for storing research data (default: ./research_data)')
    parser.add_argument('--ui', action='store_true', 
                       help='Launch the UI instead of running from command line')
    parser.add_argument('--browser-visible', action='store_true', default=True,
                      help='Show browser UI during research (default: True)')
    
    return parser.parse_args()

if __name__ == '__main__':
    try:
        # Display a welcome message
        console.print(Panel.fit(
            "[bold blue]Advanced Autonomous Research Agent[/]\n"
            "[green]A multi-agent system for methodical research using browser automation[/]",
            border_style="blue"
        ))
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Create storage path if it doesn't exist
        if not os.path.exists(args.storage_path):
            os.makedirs(args.storage_path, exist_ok=True)
        
        if args.ui or not args.query:
            # Launch UI if requested or if no query provided
            console.print("[blue]Launching user interface...[/]")
            demo = create_ui()
            demo.launch(share=True)
        else:
            # Run research from command line
            console.print(f"[blue]Starting research on query: [bold]{args.query}[/][/]")
            result = asyncio.run(run_research(
                query=args.query,
                basic_model=args.basic_model,
                advanced_model=args.advanced_model,
                max_steps=args.max_steps,
                storage_path=args.storage_path,
                browser_visible=args.browser_visible
            ))
            if result.startswith("Error:"):
                console.print(Panel(f"[bold red]Research Failed[/]\n\n{result}", border_style="red"))
                sys.exit(1)
    except ImportError as e:
        module_name = str(e).split("No module named ")[-1].strip("'")
        console.print(Panel(
            f"[bold red]Import Error:[/] {str(e)}\n\n"
            f"This error typically occurs when required Python packages are missing.\n"
            f"To fix this issue, please install the missing package:\n\n"
            f"[bold]pip install {module_name}[/]\n\n"
            f"If you're using a virtual environment, make sure it's activated first.\n"
            f"For all dependencies, run: [bold]pip install -r requirements.txt[/]",
            title="Missing Dependency",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/] {str(e)}\n\n"
            f"If this is a browser-related issue, make sure Chrome is installed and try running:\n"
            f"[bold]python start_chrome.py --verbose[/]\n\n"
            f"For more detailed logs, check the research_agent.log file.",
            title="Error Occurred",
            border_style="red"
        ))
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
