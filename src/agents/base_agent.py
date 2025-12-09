from typing import List, Dict
from openai import AzureOpenAI, RateLimitError, Omit, omit
from pydantic import BaseModel
import time

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, client: AzureOpenAI, model: str, agent_name: str):
        self.client = client
        self.model = model
        self.agent_name = agent_name
    
    def call_llm(self, messages: List[Dict], temperature: float = 0.7, max_retries: int = 8, response_format: BaseModel | Omit=omit) -> str:
        """Call Azure OpenAI API with strong retry logic."""
        delay = 5  # start with 5 seconds – Azure recommends >= 5s

        for attempt in range(max_retries):
            try:
                if response_format is omit:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                else:
                    response = self.client.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        response_format=response_format
                    )
                    return response.choices[0].message.parsed

            except Exception as e:
                # Check if it's a rate limit (Azure sometimes wraps it differently)
                is_rate_limit = (
                    isinstance(e, RateLimitError)
                    or "RateLimit" in str(e)
                    or "429" in str(e)
                    or "too many requests" in str(e).lower()
                )

                if is_rate_limit:
                    print(
                        f"[RateLimit] {self.agent_name} attempt {attempt+1}/{max_retries}. "
                        f"Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60)  # cap at 60 sec
                    continue

                # Other errors should not be silently swallowed
                print(f"[{self.agent_name}] Non-rate-limit error: {e}")
                raise e

        raise RuntimeError(f"{self.agent_name}: LLM call failed after {max_retries} retries.")
