# security.py
import re
import os
from typing import Tuple, Optional
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_TOPICS = [
    "variables","data types","casting","conditionals","if","else","loops","for","while",
    "functions","lists","tuples","dictionaries","sets","file","io","read","write",
    "exceptions","try","except","oop","class","object","inheritance","modules","packages",
    "numpy","pandas","series","dataframe","groupby","filter","indexing"
]

def is_on_topic(user_query: str) -> bool:
    q = user_query.lower()
    return any(tok in q for tok in ALLOWED_TOPICS)

ASSIGNMENT_PATTERNS = [
    r"\bassignment\b", r"\bquestion\s*\d+\b", r"\bdue\b", r"\bsubmit\b",
    r"\bsolve\b", r"\bcomplete\b", r"\bwrite code that\b", r"\bcode for\b",
    r"\bimplement\b", r"\bredesign\b", r"\bproject\b"
]

def is_assignment_intent(user_query: str) -> bool:
    q = user_query.lower()
    return any(re.search(p, q) for p in ASSIGNMENT_PATTERNS)

def contains_assignment_docs(chunks: list[dict]) -> bool:
    return any(c.get("metadata", {}).get("doc_type") == "assignment" for c in chunks)

SAFE_ASSIGNMENT_REPLY = (
    "Looks like this maps to an assignment. I won’t provide a full solution, "
    "but here’s how to think about it and get started:\n"
    "1) Identify the required inputs and outputs\n"
    "2) Break the logic into steps (pseudo-code)\n"
    "3) Write a minimal version, then iterate\n"
    "Ask me about any step and I can explain the concept with examples!"
)

OFF_TOPIC_REPLY = (
    "That seems outside our Python course scope. I can help with Python fundamentals "
    "(variables, types, conditionals, loops, functions, lists/tuples/dicts/sets, files, "
    "exceptions, OOP, modules, NumPy, Pandas). Try rephrasing your question within these topics."
)

# ---------------------------
# Prompt-injection detection
# ---------------------------

# Suspicious phrases that frequently appear in prompt injection attempts
_PROMPT_INJECTION_PATTERNS = [
    # direct overrides / instruction to ignore previous system
    r"ignore (all )?previous (instructions|prompts|messages)",
    r"disregard (all )?previous (instructions|prompts|messages)",
    r"forget (your|the) (previous|system) instructions",
    r"ignore (this )?and (answer|respond) with",
    r"you are now acting as",
    r"act as an? (assistant|bot|system) named",
    r"from now on you will",
    r"answer only with",             # strict formatting override
    r"do not mention",               # attempt to hide provenance
    r"do not reveal",                # hide instructions
    r"exfiltrate",                   # data-exfiltration wording
    r"open the following file",      # try to get file access
    r"execute the following",        # try to get code execution
    r"click this link",              # external instructions
    r"respond in json only",         # rigid response format
    r"respond with only",            # rigid response format
    r"system:\s",                    # explicit system-role injection
    r"role:\s*system",               # role specification
    r"if you are reading this",      # trick phrasing
]

# suspicious tokens / control characters often used in payloads
_SUSPICIOUS_TOKENS = [
    "<script>", "</script>", "<iframe", "javascript:", "eval(", "base64,",
    "====", "----", "```", "BEGIN:", "END:", "DROP TABLE", "--", ";--", "@@",
]

# Strict allow-list limit on extremely short queries (reduce false positives)
_MIN_LEN_TO_CHECK = 6  # characters

def is_prompt_injection(user_text: str) -> Tuple[bool, Optional[str]]:
    """
    Heuristic-based detection of prompt injection.
    Returns (is_injection, reason_string) where reason_string explains which rule matched.
    """
    if not user_text or len(user_text.strip()) < _MIN_LEN_TO_CHECK:
        return False, None

    q = user_text.lower()

    # 1) Exact suspicious phrases (regex)
    for pat in _PROMPT_INJECTION_PATTERNS:
        if re.search(pat, q):
            return True, f"Matched injection pattern: {pat}"

    # 2) Suspicious tokens
    for tok in _SUSPICIOUS_TOKENS:
        if tok in q:
            return True, f"Contained suspicious token: {tok}"

    # 3) Attempts to inject role/system blocks like "system: ... user: ..."
    #    If user message contains "system:" or "assistant:" role markers, treat as suspicious
    if re.search(r"\b(system|assistant|user)\s*:", q):
        return True, "Contains role-like markers (e.g., 'system:')."

    # 4) Attempt to smuggle instructions via quotes or long multi-part payloads mentioning 'ignore'
    if "ignore" in q and ("previous" in q or "instructions" in q or "system" in q):
        return True, "Contains explicit 'ignore previous instructions' phrasing."

    # 5) Excessive directive density (many imperative verbs in short text)
    directives = re.findall(r"\b(ignore|execute|do not|don't|follow|comply|repeat|answer|respond|print|expose|hide)\b", q)
    if len(directives) >= 3 and len(q.split()) < 80:
        return True, "Unusually high density of directives."

    return False, None

def sanitize_user_input(user_text: str) -> str:
    """
    Lightweight sanitization to remove obvious role markers and code fences.
    This does NOT make input safe against all attacks — it reduces common patterns.
    """
    text = user_text

    # remove role markers like "system:" / "assistant:" / "user:"
    text = re.sub(r"\b(system|assistant|user)\s*:\s*", " ", text, flags=re.IGNORECASE)

    # remove code fences and large separators
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"(--+|==+|__+|~~+){2,}", " ", text)

    # strip any inline script tags
    text = re.sub(r"<\s*script.*?>.*?<\s*/\s*script\s*>", " ", text, flags=re.IGNORECASE|re.DOTALL)

    return text.strip()

def chunks_contain_injection(chunks: list[dict]) -> Tuple[bool, Optional[str]]:
    """
    Inspect retrieved chunks (from vector DB) for suspicious 'system-like' instructions
    embedded in document text (sometimes PDF OCR or scraped docs include 'system:' blocks).
    Returns (bool, reason)
    """
    for c in chunks:
        txt = c.get("text", "")
        if not txt:
            continue
        low = txt.lower()
        # look for role markers in the chunk itself
        if re.search(r"\b(system|assistant|user)\s*:", low):
            return True, "Retrieved chunk contains role/system markers."
        # look for "do not reveal" / "ignore previous" in chunk (rare but possible)
        if "ignore previous" in low or "disregard prior" in low or "do not disclose" in low:
            return True, "Retrieved chunk contains directive-like wording."
    return False, None
