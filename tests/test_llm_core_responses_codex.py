"""GPT Codex / Responses API adapter tests."""

from src import llm_core


def test_codex_models_use_responses_api_on_openai_and_chatgpt():
    assert llm_core._uses_responses_api(
        "https://api.openai.com/v1/chat/completions", "gpt-5.2-codex"
    )
    assert llm_core._uses_responses_api(
        "https://chatgpt.com/backend-api/codex/responses", "gpt-5-codex"
    )


def test_build_responses_payload_converts_tool_turns():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Make a note"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1",
            "function": {"name": "create_note", "arguments": "{\"text\":\"hi\"}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "created"},
    ]
    payload = llm_core._build_responses_payload("gpt-5.2-codex", messages, 0.2, 123)

    assert payload["instructions"] == "Be concise."
    assert payload["max_output_tokens"] == 123
    assert "temperature" not in payload
    assert {"type": "function_call", "call_id": "call_1", "name": "create_note", "arguments": "{\"text\":\"hi\"}"} in payload["input"]
    assert {"type": "function_call_output", "call_id": "call_1", "output": "created"} in payload["input"]


def test_codex_payload_adds_default_instructions_for_user_only_prompt():
    payload = llm_core._build_responses_payload(
        "gpt-5.5",
        [{"role": "user", "content": "Say OK"}],
        0.0,
        5,
        stream=True,
        codex=True,
    )

    assert payload["instructions"]
    assert payload["stream"] is True
    assert "max_output_tokens" not in payload
    assert payload["input"] == [{"role": "user", "content": "Say OK"}]


def test_parse_responses_sse_text_collects_codex_stream():
    raw = '\n'.join([
        'event: response.output_text.delta',
        'data: {"type":"response.output_text.delta","delta":"O"}',
        '',
        'data: {"type":"response.output_text.delta","delta":"K"}',
        'data: {"type":"response.completed","response":{}}',
        'data: [DONE]',
    ])

    assert llm_core._parse_responses_sse_text(raw) == "OK"


def test_parse_responses_text_prefers_output_text():
    assert llm_core._parse_responses_text({"output_text": "hello"}) == "hello"
    assert llm_core._parse_responses_text({
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]
    }) == "hi"
