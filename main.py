import os
import sys
import time
import uuid
import logging
import asyncio

import gradio as gr

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from typing import List, Dict, Any, Optional, Tuple
from gradio.themes import Citrus, Default, Glass, Monochrome, Soft, Origin, Ocean, Base

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from browser_use.agent.service import Agent
from browser_use import Browser, BrowserConfig

from src.research_agent import ResearchAgent
from src.utils.chat_openrouter import ChatOpenRouter


# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="[%d-%m-%Y %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
    filename="browsing_agent.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

console = Console()

load_dotenv()


async def run_browser_task(task, max_search_iteration_input, max_query_per_iter_input, basic_provider, basic_model,
                          advanced_provider, advanced_model, llm_temperature, vision, use_own_browser, chrome_cdp, storage_path):
    """
    Run the autonomous research agent to perform a task.
    Args:
        research_task: The task to be performed.
        max_search_iteration_input: The maximum number of search iterations.
        max_query_per_iter_input: The maximum number of queries per iteration.
        provider: The provider of the language model.
        model_name: The name of the language model.
        llm_temperature: The temperature for the language model.
        use_vision: Whether to use vision capabilities.
        use_own_browser: Whether to use the user's own browser.
        chrome_cdp: The Chrome CDP configuration.
    """
    
    # Initialize LLM
    advanced_llm, advanced_error_message = initialize_llm(advanced_model, advanced_provider, llm_temperature, model_type="advanced")
    basic_llm, basic_error_message = initialize_llm(basic_model, basic_provider, llm_temperature, model_type="basic")
    if basic_error_message or advanced_error_message:
        error_message = basic_error_message if basic_error_message else advanced_error_message
        console.print(Panel(f"[red]Error initializing LLM: {error_message}[/]"))
        return
    
async def run_deep_search(task, max_search_iteration_input, max_query_per_iter_input, basic_provider, basic_model,
                          advanced_provider, advanced_model, llm_temperature, vision, use_own_browser, chrome_cdp, storage_path):
    from src.utils.deep_research import deep_research

    basic_llm, basic_error_message = initialize_llm(basic_model, basic_provider, llm_temperature, model_type="basic")
    advanced_llm, advanced_error_message = initialize_llm(advanced_model, advanced_provider, llm_temperature, model_type="advanced")
    if basic_error_message or advanced_error_message:
        error_message = basic_error_message if basic_error_message else advanced_error_message
        console.print(Panel(f"[red]Error initializing LLM: {error_message}[/]"))
        return
    markdown_content, file_path = await deep_research(task=task, llm=basic_llm, 
                                                      max_search_iterations=max_search_iteration_input,
                                                      max_query_num=max_query_per_iter_input,
                                                      use_vision=vision,
                                                      use_own_browser=use_own_browser,
                                                      chrome_cdp=chrome_cdp
                                                      )

    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)

def initialize_llm(model_name, provider, temperature: float = 0.1, model_type="basic"):
    """
    Initialize a language model with proper error handling.
    
    Args:
        model_name: The name of the model to initialize
        model_type: Whether this is a "basic" or "advanced" model (for logging)
        
    Returns:
        Tuple of (model instance, error message)
    """
    if not model_name and not provider:
        return None, f"No {model_type} model or no {provider} provider specified and no default could be determined from available API keys"
    
    try:
        # Check OpenAI models
        if provider == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None, f"OpenAI API key (OPENAI_API_KEY) is required for model: {model_name}"
            return ChatOpenAI(model=model_name, temperature=temperature, api_key=SecretStr(api_key)), None
        
        # Check Google models
        elif model_name.startswith("gemini-"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None, f"Google Gemini API key (GEMINI_API_KEY) is required for model: {model_name}"
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, api_key=SecretStr(api_key)), None
        # Use OpenRouter for any other model
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return None, f"OpenRouter API key (OPENROUTER_API_KEY) is required for model: {model_name}"
            return ChatOpenRouter(model=model_name, openai_api_key=SecretStr(api_key), temperature=temperature), None
    
    except Exception as e:
        return None, f"Error initializing {model_type} model ({model_name}): {str(e)}"

def create_ui(theme_name: str = "Monochrome") -> gr.Blocks:
    """
    Create the Gradio UI for the browser use task automation.s
    Args:
        theme_name: The name of the Gradio theme to use.
    Returns:
        A Gradio Blocks interface.
    """
    with gr.Blocks(
        title='Browser Use GUI', theme=theme_map[theme_name]
    ) as interface:
        gr.Markdown('# Browser Use Task Automation')
        with gr.Tabs() as tabs:
            with gr.TabItem("Settings", id="settings"):
                
                with gr.Row():
                    with gr.Column():
                        providers = ['Google', 'OpenRouter', 'OpenAI', 'Ollama', 'Anthropic']
                        openai_models_list = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'o4-mini', 'o3', 'o3-mini', 'gpt-4o-mini']
                        ollama_models_list = ['qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5-32b', 'qwen2.5-coder:14b', 'qwen2.5-coder:32b', 'deepseek-r1:8b', 'deepseek-r1:14b', 'deepseek-r1:32b',]
                        anthro_models_list = ['claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-7-sonnet-latest',]
                        openrouter_models_list = ['qwen/qwen2.5-vl-72b-instruct:free', 'google/gemini-2.5-pro-exp-03-25:free']
                        google_models_list = [
                            'gemini-2.0-flash', # 15 msgs per minute
                            'gemini-2.0-flash-lite', # 30 msgs per minute
                            'gemini-2.5-pro-exp-03-25', # 5 msgs per minute
                            'gemini-2.5-flash-preview-04-17', # 10 msgs per minute
                        ]

                        # Basic model initialization
                        with gr.Row():
                            basic_provider = gr.Dropdown(
                                choices=providers,
                                label='Provider for basic model',
                                value='Google',
                            )
                            
                            basic_model = gr.Dropdown(
                                choices=google_models_list,
                                label='Basic Model',
                                value=google_models_list[1],
                            )

                            basic_provider.change(
                                fn=lambda provider: gr.Dropdown(
                                    choices=(
                                        openai_models_list if provider == 'OpenAI'
                                        else ollama_models_list if provider == 'Ollama'
                                        else anthro_models_list if provider == 'Anthropic'
                                        else openrouter_models_list if provider == 'OpenRouter'
                                        else google_models_list if provider == 'Google'
                                        else []
                                    ),
                                    label='Basic Model',
                                    value=(
                                        openai_models_list[1] if provider == 'OpenAI'
                                        else ollama_models_list[1] if provider == 'Ollama'
                                        else anthro_models_list[1] if provider == 'Anthropic'
                                        else openrouter_models_list[0] if provider == 'OpenRouter'
                                        else google_models_list[1] if provider == 'Google'
                                        else ''
                                    ),
                                ),
                                inputs=basic_provider,
                                outputs=basic_model,
                            )
                            
                        # Advanced model initialization
                        with gr.Row():
                            advanced_provider = gr.Dropdown(
                                choices=providers,
                                label='Provider for advanced model',
                                value='Google',
                            )
                            
                            advanced_model = gr.Dropdown(
                                choices=google_models_list,
                                label='Advanced Model',
                                value=google_models_list[-1],
                            )

                            advanced_provider.change(
                                fn=lambda provider: gr.Dropdown(
                                    choices=(
                                        openai_models_list if provider == 'OpenAI'
                                        else ollama_models_list if provider == 'Ollama'
                                        else anthro_models_list if provider == 'Anthropic'
                                        else openrouter_models_list if provider == 'OpenRouter'
                                        else google_models_list if provider == 'Google'
                                        else []),
                                    label='Advanced Model',
                                    value=(
                                        openai_models_list[-1] if provider == 'OpenAI'
                                        else ollama_models_list[-1] if provider == 'Ollama'
                                        else anthro_models_list[-1] if provider == 'Anthropic'
                                        else openrouter_models_list[-1] if provider == 'OpenRouter'
                                        else google_models_list[-1] if provider == 'Google'
                                        else ''),
                                ),
                                inputs=advanced_provider,
                                outputs=advanced_model,
                            )
                        with gr.Row():
                            use_own_browser = gr.Checkbox(
                                label="Use Own Browser",
                                value=False,
                                info="Use your existing browser instance",
                                interactive=True
                            )
                            vision = gr.Checkbox(
                                label="Use Vision",
                                value=True,
                                info="Enable visual processing capabilities",
                                interactive=True
                            )
                        # For the deep research agents configuration
                        # TODO: Add description for each parameter
                        with gr.Row():
                            max_search_iteration_input = gr.Number(label="Max Search Iteration", value=16,
                                                                precision=0,
                                                                interactive=True,
                                                                info="Maximum number of search iterations until the research task is completed")
                            max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=8,
                                                                precision=0,
                                                                interactive=True,
                                                                info="Maximum number of search queries per iteration for as broad search results as possible")
                        
                        llm_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.6,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness in model outputs",
                            interactive=True
                        )
                        
                        storage_path = gr.Textbox(
                            label='Storage Path',
                            value='./data',
                        )
                        
                        chrome_cdp = gr.Textbox(
                            label="CDP URL",
                            placeholder="http://localhost:9222",
                            value="",
                            info="CDP for google remote debugging",
                            interactive=True,  # Allow editing only if recording is enabled
                        )
            with gr.TabItem("Chat", id="chat", ):   
                with gr.Row():
                    with gr.Column():
                        # task = gr.Textbox(
                        #     label='Task Description',
                        #     placeholder='E.g., Find flights from New York to London for next week',
                        #     lines=3,
                        # )
                        chatbot = gr.Chatbot(
                            label='Chatbot',
                            show_label=False,
                            height=600,
                            type="messages",
                        )
                        task = gr.Textbox(
                            label='Task Description',
                            placeholder='E.g., Find flights from New York to London for next week',
                            lines=3,
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button('▶️ Run Task', variant="primary")
                    
                    with gr.Column():
                        markdown_output_display = gr.Markdown(label="Research Report")
                        markdown_download = gr.File(label="Download Research Report")

        
        submit_btn.click(
                fn=run_deep_search,
                inputs=[task, max_search_iteration_input, max_query_per_iter_input, 
                        basic_provider, basic_model, advanced_provider, advanced_model, 
                        llm_temperature, vision, use_own_browser, chrome_cdp, storage_path],
                outputs=[markdown_output_display, markdown_download, submit_btn]
            )

    return interface

if __name__ == '__main__':
    console.print("[blue]Launching user interface...[/]")
    theme_name = os.getenv("THEME_NAME", "Citrus")
    interface = create_ui(theme_name)
    interface.launch(share=True)
    console.print("[blue]User interface launched successfully![/]")