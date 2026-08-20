"""Single-model adapter (ported from utils.generate_responses_in_batch)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import aiolimiter
from openai import AsyncOpenAI, BadRequestError
from tqdm.asyncio import tqdm_asyncio

from .logging import build_log_entry, log_llm_calls, print_llm_calls


class SingleModelClient:
    """Adapter for a single configured model (no routing)."""

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        gemini_thinking_level: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        # Auto-detect provider from model name if provider is the default "openai"
        # but the model name clearly belongs to another provider.
        if provider == "openai":
            lower = model_name.lower()
            if lower.startswith("claude") or lower.startswith("anthropic"):
                provider = "anthropic"
            elif lower.startswith("gemini"):
                provider = "gemini"
            elif "llama" in lower or "mistral" in lower or "mixtral" in lower:
                provider = "together"
            elif "/" in model_name:
                provider = "openrouter"
        # HAP proxy: when USE_HAP=1 is set, route openai and gemini calls
        # through the collaborator's HAP hub instead of direct API keys.
        # Anthropic stays direct (not supported by HAP).
        if os.environ.get("USE_HAP", "").strip() in ("1", "true", "yes"):
            if provider in ("openai", "gemini"):
                provider = "hap"
        self.provider = provider
        # Gemini thinking-level policy:
        #   - Caller-provided override always wins (e.g. "MINIMAL", "LOW",
        #     "MEDIUM", "HIGH"). When None, we pick a default per-model:
        #     Gemini 3 Pro reasoning models → MEDIUM (matches what the
        #     deployed Next.js validation-study app sent — utils/gemini.ts on
        #     the `nextjs` branch). All other gemini models default to
        #     MINIMAL because the IU-extraction / Phase-B JSON pipelines need
        #     deterministic, low-latency output and were the original callers.
        #   The override can also be supplied via the GEMINI_THINKING_LEVEL
        #   environment variable for ad-hoc one-off runs.
        env_override = os.environ.get("GEMINI_THINKING_LEVEL")
        if gemini_thinking_level is not None:
            self._gemini_thinking_level = gemini_thinking_level.upper()
        elif env_override:
            self._gemini_thinking_level = env_override.upper()
        elif provider == "gemini" and model_name.lower().startswith("gemini-3-pro"):
            self._gemini_thinking_level = "MEDIUM"
        else:
            self._gemini_thinking_level = "MINIMAL"

    async def _throttled_openai_chat_completion(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        n: int,
        limiter: aiolimiter.AsyncLimiter,
        reasoning_effort: Optional[str] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        _logger = logging.getLogger("sim_error")
        _empty = {"choices": [{"message": {"content": ""}} for _i in range(n)]}
        async with limiter:
            for attempt in range(3):
                try:
                    params: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                    }
                    if reasoning_effort is not None:
                        # Match H-Math production user study (nextjs branch,
                        # commit 1551fdd `utils/openai.ts`) which calls
                        # `openai.chat.completions.create({ model, messages,
                        # reasoning_effort })` WITHOUT max_completion_tokens.
                        # When reasoning_effort is set, max_completion_tokens
                        # caps the SUM of reasoning + visible-output tokens,
                        # so a low cap (caller default 3000) can be entirely
                        # consumed by reasoning on hard problems → empty
                        # assistant output. Omit the cap to let the API use
                        # its default (effectively up to the model's full
                        # output cap, e.g. 128k for gpt-5.4).
                        params["reasoning_effort"] = reasoning_effort
                    else:
                        # Non-reasoning models (e.g. gpt-4.1) still need an
                        # explicit cap to prevent runaway generation.
                        params["max_completion_tokens"] = max_tokens
                    if json_mode:
                        params["response_format"] = {"type": "json_object"}
                    # Hard wall-clock cap: hung connections must surface as a
                    # TimeoutError instead of silently blocking the whole
                    # subprocess (the SDK's internal timeout has been observed
                    # to not fire for some long reasoning-model calls).
                    return await asyncio.wait_for(
                        client.chat.completions.create(**params),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    _logger.warning("OpenAI request timed out (attempt %d/3); retrying", attempt + 1)
                    if attempt < 2:
                        await asyncio.sleep(2)
                        continue
                    _logger.error("OpenAI request exhausted retries on timeout — returning empty")
                    return _empty
                except BadRequestError as e:
                    error_body = getattr(e, "body", {}) or {}
                    error_code = error_body.get("error", {}).get("code", "") if isinstance(error_body, dict) else ""
                    if error_code == "invalid_prompt":
                        _logger.warning("Content-policy flag (attempt %d/3): %s", attempt + 1, e)
                        if attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        _logger.error("Content-policy flag: all 3 retries exhausted, returning empty response")
                        return _empty
                    raise e
                except Exception as e:
                    raise e
        return _empty

    def _map_model(self, model_name: str) -> str:
        if model_name == "gpt-4o":
            return "gpt-4o-2024-05-13"
        if model_name == "gpt-4o-241120":
            return "gpt-4o-2024-11-20"
        if model_name in ["gpt-5", "gpt-5-thinking"]:
            return "gpt-5-2025-08-07"
        if model_name in ["gpt-5-mini", "gpt-5-mini-thinking"]:
            return "gpt-5-mini-2025-08-07"
        if model_name in ["gpt-5-nano", "gpt-5-nano-thinking"]:
            return "gpt-5-nano-2025-08-07"
        # gpt-5.2 passes through unmapped — date pin can be added here once known.
        return model_name

    async def generate_responses(
        self,
        full_contexts: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        n: int = 1,
        show_progress: bool = True,
        json_mode: bool = False,
        response_schema: Optional[Any] = None,
        reasoning_effort: Optional[str] = None,
    ) -> List[List[str]]:
        """
        Generate responses for a batch of contexts using a single OpenAI model.
        Returns a list of response lists (one list per context).

        ``json_mode``: when True, ask the provider to constrain output to valid JSON.
            OpenAI uses ``response_format={"type": "json_object"}``; Gemini uses
            ``response_mime_type="application/json"`` (and ``response_schema`` if
            provided and supported by the SDK version).
        ``response_schema``: optional JSON-Schema-like dict (or provider-native
            Schema object) for strict structured output. Only honored on the
            Gemini path today; ignored on OpenAI / Anthropic.
        """
        if self.provider == "hap":
            return await self._generate_responses_hap(
                full_contexts=full_contexts,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                show_progress=show_progress,
                json_mode=json_mode,
            )

        if self.provider == "gemini":
            return await self._generate_responses_gemini(
                full_contexts=full_contexts,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                show_progress=show_progress,
                json_mode=json_mode,
                response_schema=response_schema,
            )

        if self.provider == "anthropic":
            return await self._generate_responses_anthropic(
                full_contexts=full_contexts,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                show_progress=show_progress,
            )

        # OpenAI-compatible providers: "openai", "together", "groq"
        if self.provider == "together":
            client = AsyncOpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY", ""),
                base_url="https://api.together.xyz/v1",
            )
        elif self.provider == "groq":
            client = AsyncOpenAI(
                api_key=os.environ.get("GROQ_API_KEY", ""),
                base_url="https://api.groq.com/openai/v1",
            )
        elif self.provider == "openrouter":
            client = AsyncOpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            client = AsyncOpenAI()

        # Determine reasoning effort: caller override > auto-detection.
        # When caller provides an explicit override, respect their temperature
        # too (don't force 1.0).
        _caller_reasoning = reasoning_effort
        reasoning_effort = None
        if self.model_name in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            reasoning_effort = "minimal"
            temperature = 1.0
        elif self.model_name in ["gpt-5-thinking", "gpt-5-mini-thinking", "gpt-5-nano-thinking"]:
            reasoning_effort = "medium"
            temperature = 1.0
        elif self.model_name == "gpt-5.2":
            # gpt-5.2 doesn't accept 'minimal'; supported values are
            # 'none' | 'low' | 'medium' | 'high' | 'xhigh'. Use 'low' as the
            # cost-conscious analog to 'minimal' on the gpt-5 family.
            reasoning_effort = "low"
            temperature = 1.0
        elif self.model_name == "gpt-5.4":
            # gpt-5.4's API default reasoning_effort is `none` (= no extended
            # reasoning). The H-Math production user study (nextjs branch,
            # commit 1551fdd "switch to model study", utils/openai.ts) used
            # `reasoning_effort: 'medium'` explicitly for gpt-5.4. Match that
            # so simulator-gpt-5.4 behaves like production-gpt-5.4 — otherwise
            # the model-arm comparison vs the H-Math hypothesis rankings is
            # confounded by the gpt-5.4 model running in a fundamentally
            # different mode.
            reasoning_effort = "medium"
            temperature = 1.0
        if _caller_reasoning is not None:
            reasoning_effort = _caller_reasoning

        actual_model = self._map_model(self.model_name)
        limiter = aiolimiter.AsyncLimiter(100, time_period=60)
        semaphore = asyncio.Semaphore(100)

        async def limited_task(context):
            async with semaphore:
                return await self._throttled_openai_chat_completion(
                    client=client,
                    model=actual_model,
                    messages=context,
                    temperature=temperature if temperature is not None else 0,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    n=n,
                    limiter=limiter,
                    reasoning_effort=reasoning_effort,
                    json_mode=json_mode,
                )

        async_responses = [limited_task(context) for context in full_contexts]
        _logger = logging.getLogger("sim_error")
        # Note: tqdm_asyncio.gather() doesn't support return_exceptions on older tqdm,
        # so always use asyncio.gather for reliable error isolation.
        responses = await asyncio.gather(*async_responses, return_exceptions=True)

        generated_responses: List[List[str]] = []
        for idx, resp in enumerate(responses):
            if isinstance(resp, BaseException):
                _logger.warning("Per-item API error for context %d: %s", idx, resp)
                generated_responses.append([""] * n)
                continue
            scenario_responses: List[str] = []
            for i in range(n):
                try:
                    content = resp.choices[i].message.content
                    content = content.strip()
                except Exception:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

        await log_batch_calls(
            model_name=self.model_name,
            full_contexts=full_contexts,
            outputs=generated_responses,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return generated_responses

    async def _generate_responses_hap(
        self,
        *,
        full_contexts: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        n: int,
        show_progress: bool,
        json_mode: bool = False,
    ) -> List[List[str]]:
        """Generate responses via the HAP distributed LLM hub.

        HAP proxies OpenAI and Gemini calls through a collaborator's internal
        system. Model name mapping:
          - GPT models: prepend "t-" (e.g. gpt-5.1 → t-gpt-5.1)
          - Gemini models: pass as-is
        The hub's ``hap_generate`` is synchronous, so each call is wrapped in
        ``asyncio.to_thread``. Concurrency is capped at 3 to respect hub limits.
        """
        import sys, importlib
        # Import hap_client from scripts/ directory
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        scripts_dir = os.path.normpath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        hap_client = importlib.import_module("hap_client")
        hap_generate = hap_client.hap_generate

        _logger = logging.getLogger("sim_error")

        # Map model name to HAP-expected name
        model = self.model_name
        lower = model.lower()
        if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
            model = f"t-{model}"
        elif lower == "gemini-3-flash-preview":
            model = "gemini-3-flash"

        semaphore = asyncio.Semaphore(3)

        async def run_one(context: List[Dict[str, str]]) -> List[str]:
            async with semaphore:
                results: List[str] = []
                for _ in range(n):
                    try:
                        kwargs: Dict[str, Any] = {}
                        if json_mode:
                            kwargs["is_json"] = True
                        if temperature is not None:
                            kwargs["temperature"] = temperature
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: hap_generate(
                                messages=context,
                                model=model,
                                max_tokens=max_tokens,
                                timeout=600,
                                **kwargs,
                            ),
                        )
                        text = result.get("message", "") if isinstance(result, dict) else str(result)
                        results.append(text.strip())
                    except Exception as e:
                        _logger.warning("HAP request failed: %s", e)
                        results.append("")
                return results

        tasks = [run_one(ctx) for ctx in full_contexts]
        if show_progress:
            generated_responses = list(await tqdm_asyncio.gather(*tasks))
        else:
            generated_responses = list(await asyncio.gather(*tasks))

        await log_batch_calls(
            model_name=self.model_name,
            full_contexts=full_contexts,
            outputs=generated_responses,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return generated_responses

    async def _generate_responses_anthropic(
        self,
        *,
        full_contexts: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        n: int,
        show_progress: bool,
    ) -> List[List[str]]:
        """Generate responses using the Anthropic API (Claude models)."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider selected but anthropic is not installed. "
                "Install with: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set for Anthropic provider.")

        client = AsyncAnthropic(api_key=api_key)
        limiter = aiolimiter.AsyncLimiter(40, time_period=60)
        semaphore = asyncio.Semaphore(40)
        _logger = logging.getLogger("sim_error")

        async def run_one(context: List[Dict[str, str]]) -> List[str]:
            system_text = ""
            messages = []
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_text += ("\n\n" + content if system_text else content)
                else:
                    anthropic_role = "user" if role == "user" else "assistant"
                    messages.append({"role": anthropic_role, "content": content})
            if not messages:
                return [""] * n

            async def call_once() -> str:
                # Retry transient failures (529 overload, 429 rate limit, 5xx,
                # timeouts). Without retries a single transient blip silently
                # zeros out an entire batch.
                _TEMPERATURE_DEPRECATED_PREFIXES = ("claude-opus-4-7",)
                # Per-model max_tokens override. The H-Math production user
                # study (nextjs branch, commit 1551fdd "switch to model study",
                # utils/claude.ts) sets max_tokens=4096 *literally* for
                # claude-opus-4-7. Match that exact value (not a floor) so
                # simulator behavior aligns with production. Ignores caller's
                # max_tokens for claude-opus-4-7.
                _CLAUDE_MAX_TOKENS_OVERRIDE = {"claude-opus-4-7": 4096}
                effective_max_tokens = max_tokens
                for _prefix, _val in _CLAUDE_MAX_TOKENS_OVERRIDE.items():
                    if self.model_name.startswith(_prefix):
                        effective_max_tokens = _val
                        break
                params: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": effective_max_tokens,
                }
                if system_text:
                    params["system"] = system_text
                if temperature is not None and not self.model_name.startswith(_TEMPERATURE_DEPRECATED_PREFIXES):
                    params["temperature"] = temperature
                last_err: Optional[BaseException] = None
                for attempt in range(4):
                    async with limiter:
                        async with semaphore:
                            try:
                                resp = await asyncio.wait_for(
                                    client.messages.create(**params),
                                    timeout=120.0,
                                )
                                text_blocks = [
                                    b.text for b in resp.content if hasattr(b, "text")
                                ]
                                out = "\n".join(text_blocks).strip()
                                if out:
                                    return out
                                # Empty content (no text block) — treat as
                                # transient, retry. Common with extended-thinking
                                # responses where the first block is thinking.
                                last_err = RuntimeError("anthropic returned empty content blocks")
                            except asyncio.TimeoutError as e:
                                last_err = e
                                _logger.warning("Anthropic timeout (attempt %d/4)", attempt + 1)
                            except Exception as e:
                                last_err = e
                                _logger.warning("Anthropic API error (attempt %d/4): %s", attempt + 1, e)
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
                _logger.error("Anthropic request exhausted retries: %s", last_err)
                return ""

            results: List[str] = []
            for _ in range(n):
                results.append(await call_once())
            return results

        tasks = [run_one(context) for context in full_contexts]
        if show_progress:
            generated_responses = await tqdm_asyncio.gather(*tasks)
        else:
            generated_responses = await asyncio.gather(*tasks)

        await log_batch_calls(
            model_name=self.model_name,
            full_contexts=full_contexts,
            outputs=generated_responses,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return generated_responses

    async def _generate_responses_gemini(
        self,
        *,
        full_contexts: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        n: int,
        show_progress: bool,
        json_mode: bool = False,
        response_schema: Optional[Any] = None,
    ) -> List[List[str]]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Gemini provider selected but google-genai is not installed. "
                "Install with: pip install google-genai"
            ) from exc

        api_key = (
            os.environ.get("GEMINI_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not api_key:
            raise RuntimeError(
                "GEMINI_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY must be set for Gemini provider."
            )

        client = genai.Client(api_key=api_key)

        def _map_gemini_model(model_name: str) -> str:
            if model_name.startswith("gemini-"):
                return model_name
            return "gemini-2.5-flash"

        model_name = _map_gemini_model(self.model_name)

        def _build_contents(messages: List[Dict[str, str]]) -> tuple:
            """Convert OpenAI-style messages to Gemini system_instruction + contents."""
            system_parts = []
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role == "system":
                    system_parts.append(text)
                else:
                    gemini_role = "user" if role == "user" else "model"
                    contents.append(types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=text)],
                    ))
            system_instruction = "\n\n".join(system_parts) if system_parts else None
            return system_instruction, contents

        # Gemini 3+ models have thinking enabled by default, which consumes
        # output tokens before producing the actual response. The level we want
        # depends on the use case:
        #   - IU extraction / Phase-B JSON parsing: MINIMAL (deterministic,
        #     low-latency).
        #   - Validation-study assistant calls (gemini-3-pro-preview): MEDIUM
        #     (matches what the deployed Next.js app — utils/gemini.ts on the
        #     `nextjs` branch — sent during the human study).
        # The level is configured on this client (see __init__); here we just
        # build the right ThinkingConfig defensively across SDK versions.
        thinking_level_name = self._gemini_thinking_level
        thinking_config = None
        try:
            if hasattr(types, "ThinkingLevel") and hasattr(types.ThinkingLevel, thinking_level_name):
                thinking_config = types.ThinkingConfig(
                    thinking_level=getattr(types.ThinkingLevel, thinking_level_name)
                )
            elif hasattr(types, "ThinkingLevel") and hasattr(types.ThinkingLevel, "MINIMAL"):
                # Requested level not available in this SDK build — fall back to MINIMAL.
                logging.getLogger("sim_error").warning(
                    "Gemini ThinkingLevel.%s unavailable in this SDK; falling back to MINIMAL.",
                    thinking_level_name,
                )
                thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
            else:
                # Older SDKs expose thinking_budget instead of thinking_level.
                # Map level name to a budget heuristic.
                budget = {"MINIMAL": 0, "LOW": 1024, "MEDIUM": 4096, "HIGH": 16384}.get(thinking_level_name, 0)
                thinking_config = types.ThinkingConfig(thinking_budget=budget)
        except Exception as _exc:  # pragma: no cover - best-effort
            logging.getLogger("sim_error").warning(
                "Failed to build Gemini ThinkingConfig (%s); proceeding without it.", _exc
            )
            thinking_config = None

        _gen_kwargs: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else 0,
            "max_output_tokens": max_tokens,
            "top_p": 1.0,
        }
        if thinking_config is not None:
            _gen_kwargs["thinking_config"] = thinking_config

        # Structured-output enforcement. response_mime_type is widely supported;
        # response_schema requires google-genai ≥ ~0.7. Wrap each in try/except
        # so that older SDKs degrade gracefully (plain text → caller's parser
        # still falls back to legacy regex extraction).
        if json_mode:
            try:
                _gen_kwargs["response_mime_type"] = "application/json"
            except Exception as _exc:  # pragma: no cover - best-effort
                logging.getLogger("sim_error").warning(
                    "Gemini json_mode requested but response_mime_type unsupported (%s); "
                    "falling back to plain text.", _exc
                )
        if response_schema is not None:
            try:
                _gen_kwargs["response_schema"] = response_schema
            except Exception as _exc:  # pragma: no cover - best-effort
                logging.getLogger("sim_error").warning(
                    "Gemini response_schema unsupported in this SDK version (%s); "
                    "proceeding with response_mime_type only.", _exc
                )
        try:
            generation_config = types.GenerateContentConfig(**_gen_kwargs)
        except TypeError as _exc:
            # Older SDK rejects unknown kwargs. Drop the JSON-related ones and retry.
            logging.getLogger("sim_error").warning(
                "Gemini SDK rejected JSON-mode kwargs (%s); retrying without "
                "response_mime_type / response_schema.", _exc
            )
            _gen_kwargs.pop("response_schema", None)
            _gen_kwargs.pop("response_mime_type", None)
            generation_config = types.GenerateContentConfig(**_gen_kwargs)

        _logger = logging.getLogger("sim_error")
        limiter = aiolimiter.AsyncLimiter(80, time_period=60)
        semaphore = asyncio.Semaphore(80)

        async def run_one(context: List[Dict[str, str]]) -> List[str]:
            system_instruction, contents = _build_contents(context)
            if not contents:
                return [""] * n

            config = generation_config
            if system_instruction:
                # Clone config with system_instruction. We also re-propagate the
                # JSON-mode + schema settings so they survive the rebuild.
                _clone_kwargs: Dict[str, Any] = {
                    "temperature": generation_config.temperature,
                    "max_output_tokens": generation_config.max_output_tokens,
                    "top_p": generation_config.top_p,
                    "system_instruction": system_instruction,
                }
                if thinking_config is not None:
                    _clone_kwargs["thinking_config"] = thinking_config
                if json_mode:
                    _clone_kwargs["response_mime_type"] = "application/json"
                if response_schema is not None:
                    _clone_kwargs["response_schema"] = response_schema
                try:
                    config = types.GenerateContentConfig(**_clone_kwargs)
                except TypeError:
                    # Same defensive degradation as above.
                    _clone_kwargs.pop("response_schema", None)
                    _clone_kwargs.pop("response_mime_type", None)
                    config = types.GenerateContentConfig(**_clone_kwargs)

            async def call_once() -> str:
                async with limiter:
                    async with semaphore:
                        try:
                            # Hard wall-clock cap to prevent hung Gemini
                            # connections from stalling the whole subprocess.
                            response = await asyncio.wait_for(
                                client.aio.models.generate_content(
                                    model=model_name,
                                    contents=contents,
                                    config=config,
                                ),
                                timeout=120.0,
                            )
                            return (response.text or "").strip()
                        except asyncio.TimeoutError:
                            _logger.warning("Gemini request timed out after 120s")
                            return ""
                        except Exception as e:
                            _logger.warning("Gemini API error: %s", e)
                            return ""

            async def call_with_retry() -> str:
                retries = 2
                for attempt in range(retries + 1):
                    text = await call_once()
                    if text:
                        return text
                    if attempt < retries:
                        await asyncio.sleep(0.8)
                return ""

            if n == 1:
                return [await call_with_retry()]

            results: List[str] = []
            for _ in range(n):
                results.append(await call_with_retry())
            return results

        tasks = [run_one(context) for context in full_contexts]
        if show_progress:
            generated_responses = await tqdm_asyncio.gather(*tasks)
        else:
            generated_responses = await asyncio.gather(*tasks)

        await log_batch_calls(
            model_name=self.model_name,
            full_contexts=full_contexts,
            outputs=generated_responses,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return generated_responses


async def log_batch_calls(
    *,
    model_name: str,
    full_contexts: List[List[Dict[str, str]]],
    outputs: List[List[str]],
    temperature: float,
    max_tokens: int,
    n: int,
) -> None:
    log_entries = []
    for context, out in zip(full_contexts, outputs):
        system_prompt = ""
        user_prompt = ""
        for msg in context:
            if msg.get("role") == "system" and not system_prompt:
                system_prompt = msg.get("content", "")
        for msg in reversed(context):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
        log_entries.append(
            build_log_entry(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=context,
                output=out,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
        )
    await log_llm_calls(log_entries)
    print_llm_calls(log_entries)

