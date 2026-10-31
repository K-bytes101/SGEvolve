import os
import time
import requests
import json
import logging
from pathlib import Path
from typing import Optional

# 配置 logger
def _setup_logger():
    """Configure logger, write logs to log/request.log"""
    # Get project root directory (assuming current file is in model/ directory)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    log_dir = project_root / 'log'
    
    # Ensure log directory exists
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'request.log'
    
    # Create logger
    logger = logging.getLogger('LLMRequester')
    logger.setLevel(logging.INFO)
    
    # Avoid adding handler repeatedly
    if logger.handlers:
        return logger
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

# Get configured logger
logger = _setup_logger()

MAX_RETRIES = 3

class LLMRequester:
    """
    LLM Request logic wrapper class
    Supports custom model, Prompt, timeout setting and error handling
    """
    def __init__(self, api_key: str, base_url: str = "", seed: int = 42):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = 240
        self.seed = seed

    # Gemini series model wrapper
    def gemini_response(self, prompt: str, model: str, max_tokens: int = 2048, system_prompt: Optional[str] = None):
        url = f"{self.base_url}/v1beta/models/{model}:generateContent"

        # Check if the model supports reasoning
        reasoning = "nothinking" not in model.lower()

        payload = {}
        # System instruction
        if system_prompt is not None:
            payload["system_instruction"] = {
                "role": "system",
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            }
        # Content
        payload["contents"] = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        # Generation configuration
        if reasoning:
            payload["generationConfig"] = {
                "maxOutputTokens": max_tokens,
                "thinking_config": {
                    "include_thoughts": False,
                    "thinking_level": "medium"
                }
            }
        else:
            payload["generationConfig"] = {
                "maxOutputTokens": max_tokens,
            }

        # Send request
        try:
            response = requests.post(
                url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )

            # Check HTTP status code
            response.raise_for_status()

            result = response.json()
            # Extract text content
            content = result["candidates"][0]["content"]["parts"][0]["text"]

            return content
        except requests.exceptions.Timeout:
            logger.error("Request timeout, please check the network or model service status.")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return None


    def gpt_response(self, prompt: str, model: str, max_tokens: int = 2048, system_prompt: Optional[str] = None):
        url = f"{self.base_url}/v1/chat/completions"

        # Construct standard OpenAI format Payload
        if system_prompt is None:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "seed": self.seed,
                "n": 1,
                "reasoning_effort": "medium"
            }
        else:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "seed": self.seed,
                "n": 1,
                "reasoning_effort": "medium"
            }

        try:
            logger.info(f"Requesting model: {model} ...")
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            # Check HTTP status code
            response.raise_for_status()

            result = response.json()
            # Extract text content
            content = result["choices"][0]["message"]["content"]
            return content
        except requests.exceptions.Timeout:
            logger.error("Request timeout, please check the network or model service status.")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return None


    def get_response(self, prompt: str, model: str, max_tokens: int = 2048, system_prompt: Optional[str] = None):
        """
        Send request and get response result; retry (exponential backoff interval) when the underlying returns None.
        """
        if model.startswith("gemini-"):
            caller = self.gemini_response
        elif model.startswith("gpt-"):
            caller = self.gpt_response

        else:
            logger.error(f"Unsupported model: {model}")
            return None

        for attempt in range(1, MAX_RETRIES + 1):
            result = caller(prompt, model, max_tokens, system_prompt)
            if result is not None:
                return result
            if attempt < MAX_RETRIES:
                wait = min(2 ** (attempt - 1), 16)
                logger.warning(
                    f"Model {model} request failed, retry after {wait}s (total {MAX_RETRIES} attempts)..."
                )
                time.sleep(wait)

        logger.error(f"Model {model} still not returned valid content after {MAX_RETRIES} attempts")
        return None


if __name__ == '__main__':
    base_url = "" or os.getenv("BASE_URL")
    if not base_url:
        raise Exception("BASE_URL environment variable is not set")

    api_key = os.getenv("API_KEY")
    if  not api_key:
        raise Exception("API_KEY environment variable is not set")
    llm = LLMRequester(api_key=api_key, base_url=base_url)

    response = llm.get_response(prompt="1 + 1 = ?", model="gemini-3-flash-preview")
    print('response', response)
