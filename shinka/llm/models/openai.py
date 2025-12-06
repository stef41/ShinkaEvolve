import backoff
import openai
import re
from .pricing import OPENAI_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def clean_model_output(content: str) -> str:
    """Clean special tokens from model output."""
    if not content:
        return content
    
    # For gpt-oss-120b: Extract content from the 'final' channel specifically
    # Format: <|channel|>final<|message|>...actual content...<|end|> or just ends
    if '<|channel|>final<|message|>' in content:
        # Extract from final channel
        pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    elif '<|message|>' in content:
        # Fallback: extract all <|message|> blocks and use the last one
        pattern = r'<\|message\|>(.*?)(?:<\|end\|>|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            # Use the last message block (skip reasoning)
            content = matches[-1].strip()
    
    # Remove any remaining special tokens
    content = re.sub(r'<\|[^>]+\|>', '', content)
    
    return content.strip()


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
    giveup=lambda e: isinstance(e, openai.InternalServerError) and 'channel' in str(e).lower(),
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    # Use standard chat completions for:
    # 1. Custom models (those with "/" in name like "openai/gpt-oss-120b")
    # 2. Models not in OPENAI_MODELS (custom base URL endpoints like vLLM)
    is_custom_model = "/" in model or model not in OPENAI_MODELS
    
    if is_custom_model:
        # Map max_output_tokens to max_tokens for standard chat completions API
        api_kwargs = kwargs.copy()
        if "max_output_tokens" in api_kwargs:
            api_kwargs["max_tokens"] = api_kwargs.pop("max_output_tokens")
        
        # Ensure sufficient tokens for code generation (gpt-oss-120b needs more)
        if "max_tokens" not in api_kwargs:
            api_kwargs["max_tokens"] = 4096  # Default to 4k for code generation
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    *new_msg_history,
                ],
                **api_kwargs,
            )
            raw_content = response.choices[0].message.content or ""
            logger.info(f"RAW LLM OUTPUT (first 500 chars): {raw_content[:500]}")
            # Clean special tokens from output
            content = clean_model_output(raw_content)
            logger.info(f"CLEANED OUTPUT (first 500 chars): {content[:500]}")
            new_msg_history.append({"role": "assistant", "content": content})
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            # Return empty content on error
            content = ""
            new_msg_history.append({"role": "assistant", "content": content})
            input_tokens = 0
            output_tokens = 0
    elif output_model is None:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            content = response.output[0].content[0].text
        except Exception:
            # Reasoning models - ResponseOutputMessage
            content = response.output[1].content[0].text
        new_msg_history.append({"role": "assistant", "content": content})
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
    else:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            text_format=output_model,
            **kwargs,
        )
        content = response.output_parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    # Calculate costs (use 0 for custom models not in pricing table)
    if model in OPENAI_MODELS:
        input_cost = OPENAI_MODELS[model]["input_price"] * input_tokens
        output_cost = OPENAI_MODELS[model]["output_price"] * output_tokens
    else:
        # Custom endpoint model - no pricing available
        input_cost = 0.0
        output_cost = 0.0
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result
