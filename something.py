import os
import asyncio 

import gradio as gr

from rich.text import Text
from rich.panel import Panel
from rich.console import Console

from rich import print
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_core.utils.utils import secret_from_env

from typing import Optional, List
from pydantic import Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig


load_dotenv()

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("OPENROUTER_API_KEY", default=None)
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url=os.getenv('OPENROUTER_API_URL'), openai_api_key=openai_api_key, **kwargs)

@dataclass
class ActionResult:
    is_done: bool
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False

@dataclass
class AgentHistoryList:
    all_results: List[ActionResult]
    all_model_outputs: List[dict]
    
def parse_agent_history(history_str: str) -> None:
    console = Console()
    
    print(history_str)

    sections = history_str.split('ActionResult(')
    for i, section in enumerate(sections[1:], start=1):
        content = ''
        if 'extracted_content=' in section:
            content = section.split('extracted_content=')[1].split(',')[0].strip("'")

        if content: 
            header = Text(f"Steppp {i}", style="bold magenta")
            panel = Panel(
                content,
                title=header,
                border_style="green",
            )
            console.print(panel)
            console.print() 
            
async def run_browser_task(
    task: str,
    model: str,
    provider: str,
):
    if not model:
        raise ValueError('Model is required')
    if not task:
        raise ValueError('Task is required')
    if not task.strip():
        raise ValueError('Task cannot be empty')

    try:
        if provider == 'Google':
            llm = ChatGoogleGenerativeAI(
                model=model,
                api_key=SecretStr(os.getenv('GEMINI_API_KEY')),
            )
        elif provider == 'OpenRouter':
            llm = ChatOpenRouter(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        browser = Browser(
            config=BrowserConfig( 
                new_context_config=BrowserContextConfig(
                    viewport_expansion=0,
                )
            )
        )
        
        agent = Agent(
            task=task,
            llm=llm,
            max_actions_per_step=8,
            browser=browser,
        )

        result = await agent.run(max_steps=111)
        print(result)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def create_ui():
	with gr.Blocks(title='Browser Use GUI') as interface:
		gr.Markdown('# Browser Use Task Automation')

		with gr.Row():
			with gr.Column():
				task = gr.Textbox(
					label='Task Description',
					placeholder='E.g., Find flights from New York to London for next week',
					lines=3,
				)
				model = gr.Dropdown(
					choices=['gemini-2.0-flash', 'gemini-2.5-flash-preview-04-17'],
					label='Model',
					value='gemini-2.5-flash-preview-04-17',
				)

				provider = gr.Dropdown(
					choices=['Google', 'OpenRouter', 'DeepSeek'],
					label='Provider',
					value='Google',
				)
				provider.change(
					fn=lambda provider: gr.Dropdown(
						choices=(
							['gemini-2.0-flash', 'gemini-2.5-flash-preview-04-17'] if provider == 'Google'
							else ['qwen/qwen2.5-vl-72b-instruct:free', 'google/gemini-2.5-pro-exp-03-25:free'] if provider == 'OpenRouter'
							else ['deepseek/deepseek-chat-v3-0324:free']
						),
						label='Model',
						value=(
							'gemini-2.5-flash-preview-04-17' if provider == 'Google'
							else 'qwen/qwen2.5-vl-72b-instruct:free' if provider == 'OpenRouter'
							else 'deepseek/deepseek-chat-v3-0324:free'
						),
					),
					inputs=provider,
					outputs=model,
				)


					# model = gr.Dropdown( 
					#     choices=['qwen/qwen2.5-vl-72b-instruct:free', 'gemini-2.0-flash', 'google/gemini-2.5-pro-exp-03-25:free', 'deepseek/deepseek-chat-v3-0324:free'], 
					#     label='Model', 
					#     value='qwen/qwen2.5-vl-72b-instruct:free',
					# )
				submit_btn = gr.Button('Run Task')

			with gr.Column():
				output = gr.Textbox(label='Output', lines=10, interactive=False)

		submit_btn.click(
			fn=lambda *args: asyncio.run(run_browser_task(*args)),
			inputs=[task, model, provider],
			outputs=output,
		)

	return interface


if __name__ == '__main__':
	demo = create_ui()
	demo.launch(share=True)
