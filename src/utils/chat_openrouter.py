import os

from typing import Optional, Union
from pydantic import Field, SecretStr
from langchain_openai import ChatOpenAI

class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 openai_api_key: SecretStr,
                 **kwargs):
        super().__init__(base_url=os.getenv('OPENROUTER_API_URL'), api_key=openai_api_key, **kwargs)
    
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

