import os

import pipmaster as pm

if not pm.is_installed("openai"):
    pm.install("openai")

import numpy as np
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)
from lightrag.llm.openai import openai_complete_if_cache
from typing import Any, Union
from collections.abc import AsyncIterator

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


async def deepinfra_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using DeepInfra's OpenAI-compatible API.

    Reads the model name from LightRAG's global config (llm_model_name).
    Set DEEPINFRA_API_KEY in your environment.
    """
    if history_messages is None:
        history_messages = []
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        base_url=DEEPINFRA_BASE_URL,
        api_key=api_key,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
    model_name="BAAI/bge-large-en-v1.5",
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def deepinfra_embed(
    texts: list[str],
    model: str = "BAAI/bge-large-en-v1.5",
    api_key: str | None = None,
) -> np.ndarray:
    """Generate embeddings using DeepInfra's OpenAI-compatible embeddings endpoint.

    Default model: BAAI/bge-large-en-v1.5 (1024-dim).
    Set DEEPINFRA_API_KEY in your environment, or pass api_key explicitly.
    """
    if api_key is None:
        api_key = os.environ.get("DEEPINFRA_API_KEY")

    client = AsyncOpenAI(base_url=DEEPINFRA_BASE_URL, api_key=api_key)
    async with client:
        response = await client.embeddings.create(model=model, input=texts)
    return np.array([dp.embedding for dp in response.data])
