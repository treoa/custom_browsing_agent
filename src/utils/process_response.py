import os
import json
import logging

import gradio as gr

from typing import Tuple
from rich.console import Console
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


logger = logging.getLogger(__name__)

async def process_user_response(user_response, chatbot, basic_llm: BaseChatModel, advanced_llm: BaseChatModel):
    global global_agent, conversation_history, current_plan, plan_approved

    if not current_plan:
        chatbot.append((user_response, "No plan to approve or modify."))
        return chatbot

    # Use LLM to classify user response
    classification, reason = await classify_user_response(user_response, current_plan, basic_llm)

    if classification == "approval":
        plan_approved = True
        chatbot.append((user_response, "Plan approved. Executing..."))
        conversation_history.append((user_response, "Plan approved. Executing..."))

        # Execute the plan
        history = await global_agent.run(max_steps=100)
        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        chatbot.append(("", f"Final Result: {final_result}"))
        conversation_history.append(("", f"Final Result: {final_result}"))

        return chatbot
    elif classification == "modification":
        chatbot.append((user_response, f"Modifying plan based on user feedback: {reason}"))
        conversation_history.append((user_response, f"Modifying plan based on user feedback: {reason}"))
        modified_plan = await global_agent.modify_plan(reason)
        current_plan = modified_plan
        chatbot.append(("", f"Modified Plan: {modified_plan}"))
        conversation_history.append(("", f"Modified Plan: {modified_plan}"))
        return chatbot
    else:  # Unclear
        chatbot.append((user_response, "I'm not sure what you mean. Please clarify your response."))
        conversation_history.append((user_response, "I'm not sure what you mean. Please clarify your response."))
        return chatbot
    
async def classify_user_response(user_response: str, current_plan: str, llm: BaseChatModel) -> Tuple[str, str]:
    """
    Classifies the user's response into 'approval', 'modification', 'rejection', or 'unclear'.

    Args:
        user_response: The user's response string.
        current_plan: The current plan string.
        llm: The language model to use for classification.

    Returns:
        A tuple containing the classification ('approval', 'modification', 'rejection', 'unclear') and a reason string.
    """
    system_prompt = """
    You are a helpful assistant that classifies user response's attitude to a given plan in terms of readiness to execute it fully, or to modify it.
    Your task is to determine if the user approves the whole plan to start approach on that, or user wants to make some changes to it.
    Respond with a JSON object with the following format:
    {{
        "classification": "approval" | "disapproval",
    }}
    """

    human_prompt = """
    Current Plan: {current_plan}
    User Response: {user_response}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({"current_plan": current_plan, "user_response": user_response})

    try:
        response_json = json.loads(response)
        classification = response_json.get("classification", "unclear")
        reason = response_json.get("reason", "")
        return classification, reason
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response: {response}")
        return "disapproval", "Could not understand the response."