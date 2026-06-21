"""
Arize AX tracing setup. Must be imported before any `anthropic` import.

Usage:
    import instrumentation          # at the top of your entrypoint
    instrumentation.setup()

    # ... run your app ...

    instrumentation.shutdown()      # flush spans before process exits (CLI scripts)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_tracer_provider = None


def setup():
    global _tracer_provider

    space_id = os.environ.get("ARIZE_SPACE_ID", "")
    api_key = os.environ.get("ARIZE_API_KEY", "")
    project_name = os.environ.get("ARIZE_PROJECT_NAME", "attention-retention")

    if not space_id or not api_key:
        print("WARNING: ARIZE_SPACE_ID / ARIZE_API_KEY not set — tracing disabled.")
        return None

    try:
        from arize.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        _tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
        )
        AnthropicInstrumentor().instrument(tracer_provider=_tracer_provider)
        print(f"Arize tracing enabled -> project: {project_name}")
    except Exception as e:
        print(f"WARNING: Arize tracing setup failed: {e}")

    return _tracer_provider


def shutdown():
    if _tracer_provider:
        _tracer_provider.force_flush()
        _tracer_provider.shutdown()
