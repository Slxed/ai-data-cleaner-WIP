"""One function that posts a chat request to an OpenAI-compatible API.

config.py points this at Gemini unless LLM_BASE_URL or LLM_MODEL is set.
"""

import json
import urllib.error
import urllib.request

from config import LLM_BASE_URL, LLM_MODEL, llm_api_key


class LLMError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def message_text(message):
    """Plain text from an assistant message. Tool calls are separate."""
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "\n".join(parts)
    return str(content)


def chat(messages, tools=None, tool_name=None):
    """Send messages and return the assistant message dict.

    When tool_name is set, we ask the model to call that tool. If the server
    rejects that request, we try once more with tool_choice "auto".
    """
    key = llm_api_key()
    if not key:
        raise LLMError("No API key is set.")

    body = {"model": LLM_MODEL, "messages": messages}
    # Gemini 3 is tuned for its default sampling. Sending temperature can
    # make the model loop, so we leave it out on purpose.
    if tools:
        body["tools"] = tools
        if tool_name:
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_name},
            }
        else:
            body["tool_choice"] = "auto"

    try:
        return _post(body, key)
    except LLMError as err:
        if tools and tool_name and err.status == 400:
            body["tool_choice"] = "auto"
            return _post(body, key)
        raise


def _post(body, key):
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        detail = detail.replace(key, "[redacted]")[:500]
        raise LLMError(
            f"LLM request failed ({err.code}): {detail}",
            status=err.code,
        ) from None
    except urllib.error.URLError as err:
        raise LLMError(f"Could not reach the LLM API: {err.reason}") from None

    try:
        payload = json.loads(raw)
        message = payload["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        raise LLMError("The LLM response did not include a message I could read.") from None
    if not isinstance(message, dict):
        raise LLMError("The LLM response did not include a message I could read.")
    return message
