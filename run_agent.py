import os
import glob

import gradio as gr

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils import utils
from src.utils.utils import MissingAPIKeyError
from src.utils.chat_openrouter import ChatOpenRouter

load_dotenv()


def initialize_llm(model_name, provider, model_type="basic"):
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
            return ChatOpenAI(model=model_name, temperature=0.1), None
        
        # Check Google models
        elif model_name.startswith("gemini-"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None, f"Google Gemini API key (GEMINI_API_KEY) is required for model: {model_name}"
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.1, api_key=SecretStr(api_key)), None
        # Use OpenRouter for any other model
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return None, f"OpenRouter API key (OPENROUTER_API_KEY) is required for model: {model_name}"
            return ChatOpenRouter(model=model_name, openai_api_key=SecretStr(api_key)), None
    
    except Exception as e:
        return None, f"Error initializing {model_type} model ({model_name}): {str(e)}"

async def run_browser_agent(task, max_search_iteration_input, max_query_per_iter_input, basic_provider, basic_model,
                          advanced_provider, advanced_model, llm_temperature, vision, use_own_browser, chrome_cdp, chatbot):
    try:
        advanced_llm, error_message = initialize_llm(advanced_model, advanced_provider, model_type="advanced")
        basic_llm, error_message = initialize_llm(basic_model, basic_provider, model_type="basic")
        
        try:
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        except exception as e:
                raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        # latest_video = None
        # if save_recording_path:
        #     new_videos = set(
        #         glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
        #         + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
        #     )
        #     if new_videos - existing_videos:
        #         latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )

    except MissingAPIKeyError as e:
        logger.error(str(e))
        raise gr.Error(str(e), print_exception=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',  # final_result
            errors,  # errors
            '',  # model_actions
            '',  # model_thoughts
            None,  # latest_video
            None,  # history_file
            None,  # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )