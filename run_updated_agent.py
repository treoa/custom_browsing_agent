#!/usr/bin/env python3
"""
Run Updated Research Agent Script

This script provides a simple command-line interface for running the updated
research agent with improved browser-use integration.
"""

import os
import sys
import asyncio
import argparse
import time
import json
from pathlib import Path
import logging

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.research_agent_updated import ResearchAgent
from src.utils.project_logger import ProjectLogger
from src.utils.browser_config import BrowserConfigUtils
from src.adapter import run_research_with_updated_agent

# Load environment variables
load_dotenv()

# Configure console for rich output
console = Console()

def get_language_model(model_name):
    """
    Get a language model instance based on the model name.
    
    This function is adapted from main.py to maintain compatibility.
    
    Args:
        model_name: Model name/identifier
        
    Returns:
        Language model instance
    """
    # Import here to avoid circular imports
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.utils.utils import secret_from_env
    from pydantic import Field, SecretStr
    
    if model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=SecretStr(os.getenv('GEMINI_API_KEY')),
        )
    elif model_name.startswith("gpt"):
        if os.getenv('OPENROUTER_API_KEY'):
            # Import dynamically from main.py
            from main import ChatOpenRouter
            return ChatOpenRouter(model=model_name)
        else:
            return ChatOpenAI(
                model=model_name,
                api_key=os.getenv('OPENAI_API_KEY'),
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

async def main():
    parser = argparse.ArgumentParser(description='Run updated research agent with improved browser-use integration')
    parser.add_argument('--query', type=str, help='Research query to investigate')
    parser.add_argument('--basic-model', type=str, default='gemini-2.0-flash', 
                       help='LLM model to use for basic tasks (default: gemini-2.0-flash)')
    parser.add_argument('--advanced-model', type=str, default='gemini-2.5-pro-exp-03-25', 
                       help='LLM model to use for complex tasks (default: gemini-2.5-pro-exp-03-25)')
    parser.add_argument('--max-steps', type=int, default=50,
                       help='Maximum number of research steps (default: 50)')
    parser.add_argument('--storage-path', type=str, default='./research_data',
                       help='Path for storing research data (default: ./research_data)')
    parser.add_argument('--browser-visible', action='store_true', default=True,
                      help='Show browser UI during research (default: True)')
    parser.add_argument('--no-browser-visible', action='store_false', dest='browser_visible',
                      help='Hide browser UI during research')
    parser.add_argument('--use-vision', action='store_true', default=True,
                      help='Use vision capabilities (default: True)')
    parser.add_argument('--no-vision', action='store_false', dest='use_vision',
                      help='Disable vision capabilities')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('research_agent.log')
        ]
    )
    
    # Use default query for demo if not provided
    if not args.query:
        args.query = "What are the latest developments in quantum computing and their potential applications?"
        console.print(f"[yellow]No query provided, using default: [bold]{args.query}[/bold][/yellow]")
    
    # Display configuration
    console.print(Panel(
        f"[bold blue]Research Configuration[/]\n\n"
        f"Query: {args.query}\n"
        f"Basic Model: {args.basic_model}\n"
        f"Advanced Model: {args.advanced_model}\n"
        f"Max Steps: {args.max_steps}\n"
        f"Storage Path: {args.storage_path}\n"
        f"Browser Visible: {args.browser_visible}\n"
        f"Use Vision: {args.use_vision}\n"
        f"Log Level: {args.log_level}",
        title="Configuration",
        border_style="blue"
    ))
    
    # Prompt to continue
    console.print("\nPress Enter to start research, or Ctrl+C to cancel...", end="")
    try:
        input()
    except KeyboardInterrupt:
        console.print("\n[bold red]Research cancelled by user[/]")
        return

    # Set up progress display
    progress = Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    )
    
    # Progress callback function
    def update_progress(progress_percentage, total, description):
        nonlocal task
        progress.update(task, completed=progress_percentage, description=f"[bold blue]{description}[/]")

    # Create storage path if it doesn't exist
    os.makedirs(args.storage_path, exist_ok=True)

    # Start timer
    start_time = time.time()
    
    # Run research with progress display
    with progress:
        # Add task for progress tracking
        task = progress.add_task("[bold blue]Starting research...[/]", total=100)
        
        try:
            # Run research
            result = await run_research_with_updated_agent(
                query=args.query,
                basic_model=args.basic_model,
                advanced_model=args.advanced_model,
                max_steps=args.max_steps,
                storage_path=args.storage_path,
                browser_visible=args.browser_visible,
                use_vision=args.use_vision
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Display success
            console.print(Panel(
                f"[bold green]Research completed in {duration:.2f} seconds![/]",
                title="Success",
                border_style="green"
            ))
            
            # Display results
            if "output" in result and isinstance(result["output"], dict) and "content" in result["output"]:
                console.print(Panel(
                    result["output"]["content"][:1000] + "..." if len(result["output"]["content"]) > 1000 else result["output"]["content"],
                    title="Research Summary (Preview)",
                    border_style="blue"
                ))
            else:
                console.print(Panel(
                    str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result),
                    title="Research Results (Raw)",
                    border_style="yellow"
                ))
            
            # Display output location
            console.print(f"\n[bold green]Full results available in:[/] {args.storage_path}")
        
        except Exception as e:
            # Display error
            import traceback
            console.print(Panel(
                f"[bold red]Error:[/] {str(e)}\n\n{traceback.format_exc()}",
                title="Research Failed",
                border_style="red"
            ))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Research cancelled by user[/]")
        sys.exit(1)
