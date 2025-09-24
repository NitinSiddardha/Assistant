# utils.py
import os
import re
import json
import logging
from typing import Dict, Tuple

# Google Gemini integration
import google.generativeai as genai
# OpenAI integration
from openai import OpenAI
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from dotenv import load_dotenv

load_dotenv()



# API Keys (optional)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HAS_GEMINI = bool(GOOGLE_API_KEY)
HAS_OPENAI = bool(OPENAI_API_KEY)

if HAS_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logging.warning("GOOGLE_API_KEY not set. Gemini calls will be skipped.")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if HAS_OPENAI else None
if not HAS_OPENAI:
    logging.warning("OPENAI_API_KEY not set. OpenAI calls will be skipped.")

# Model names (only valid ones)
FAST_MODEL = os.getenv("FAST_MODEL_ID", "gemini-1.5-flash")
BIG_MODEL = os.getenv("BIG_MODEL_ID", "gpt-4o")

# Thresholds and params
PROMPT_LENGTH_THRESHOLD = int(os.getenv("PROMPT_LENGTH_THRESHOLD", "400"))
MAX_TOKENS_UNIFIED = int(os.getenv("MAX_TOKENS_UNIFIED", "1024"))

# spaCy pipeline
try:
    nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except Exception:
    nlp = None
    _HAS_SPACY = False

# --- Helpers ---

def get_api_type(model_name: str) -> str:
    """Determines API type ('openai' or 'gemini' or 'anthropic') from model name."""
    model_lower = model_name.lower()
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    elif "gemini" in model_lower:
        return "gemini"
    elif "claude" in model_lower:
        return "anthropic"
    return "unknown"

def _resp_text(resp) -> str:
    """Safely extracts text from API response."""
    return getattr(resp, "text", str(resp))

def _extract_json_block(text: str) -> Dict:
    """Robustly finds and parses the first JSON block from a string."""
    if not text:
        return {}

    cleaned = re.sub(r"^```[a-zA-Z]*", "", text.strip(), flags=re.M)
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if m:
        block = m.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            try:
                return json.loads(block.replace("'", '"'))
            except Exception:
                return {}
    return {}

# --- Core Logic ---

UNIFIED_TEMPLATE = (
    "You are a helpful assistant. First, analyze the user's prompt to assess its complexity and task type. "
    "Second, provide a direct and helpful response to the prompt. "
    "Return ONLY a single valid JSON object with no extra text, markdown, or explanation. "
    "The JSON object must have two keys: 'assessment' and 'response'.\n"
    "The 'assessment' object must contain: {{'complexity': <float 0.0-1.0>, 'task_type': '<one of: Open QA, Closed QA, Summarization, Text Generation, Code Generation, Chatbot, Classification, Rewrite, Brainstorming, Extraction, Other>'}}.\n"
    "The 'response' key must contain your full text answer to the user's prompt.\n"
    "User Prompt: \n\"\"\"{prompt}\"\"\"\n"
    "Return JSON:"
)

def choose_model_heuristic(prompt: str) -> Tuple[str, str]:
    """Chooses a model based on prompt length only (no API call)."""
    if len(prompt) > PROMPT_LENGTH_THRESHOLD:
        return BIG_MODEL, "Big Model - routed due to long prompt"
    return FAST_MODEL, "Fast Model - routed due to short prompt"

# ---- Multi-model failover ----
MODEL_CHAIN = [
    "gpt-4o",            # OpenAI big model
    "gpt-4o-mini",       # OpenAI medium
    "gpt-3.5-turbo",     # OpenAI fast
    "gemini-1.5-flash",  # Google fast
    "gemini-1.5-pro",    # Google big
    "claude-3-opus",     # Anthropic big
    "claude-3-haiku",    # Anthropic fast
]

def _call_model(model_name: str, prompt: str) -> Tuple[Dict, str]:
    """Single model API call with JSON parsing."""
    api_type = get_api_type(model_name)
    formatted_prompt = UNIFIED_TEMPLATE.format(prompt=prompt)
    raw_response = ""

    if api_type == "openai":
        if not HAS_OPENAI or openai_client is None:
            raise RuntimeError("OpenAI not configured")
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=MAX_TOKENS_UNIFIED,
            temperature=0.7
        )
        raw_response = completion.choices[0].message.content

    elif api_type == "gemini":
        if not HAS_GEMINI:
            raise RuntimeError("Gemini not configured")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            formatted_prompt,
            generation_config={"max_output_tokens": MAX_TOKENS_UNIFIED}
        )
        raw_response = _resp_text(resp)

    elif api_type == "anthropic":
        # Placeholder: add Anthropic client integration here if you have keys
        raise NotImplementedError("Anthropic API integration not yet implemented")

    parsed_json = _extract_json_block(raw_response)
    assessment = parsed_json.get(
        "assessment",
        {"complexity": 0.5, "task_type": "Other", "error": "parsing_failed"}
    )
    response = parsed_json.get(
        "response",
        "Error: Failed to parse the response from the model."
    )
    return assessment, response

def unified_call(model_name: str, prompt: str) -> Tuple[Dict, str]:
    """
    Tries multiple models in sequence until one works.
    Returns (assessment, response).
    """
    for candidate in [model_name] + [m for m in MODEL_CHAIN if m != model_name]:
        try:
            return _call_model(candidate, prompt)
        except Exception as e:
            logging.warning(f"[DEBUG] {candidate} failed: {e}, trying next...")

    return (
        {"complexity": 0.0, "task_type": "Error", "error": "all_models_failed"},
        "❌ Sorry, all models are unavailable right now. Please try again later."
    )
