"""
Archon Input/Output Sanitization Module

Implements security controls for LLM inputs and outputs.
Based on OWASP Agentic AI security guidelines.
"""

import re
from typing import Optional
from dataclasses import dataclass

from src.core.errors import PromptInjectionError, ValidationError
from src.monitoring.logger import get_logger

logger = get_logger("security.sanitization")


# =============================================================================
# Prompt Injection Detection Patterns
# =============================================================================

# Common prompt injection patterns
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all|what)\s+(you|i)\s+(know|said|told)",

    # Role manipulation
    r"you\s+are\s+(now|actually)\s+(a|an|the)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if|though)",
    r"roleplay\s+as",
    r"simulate\s+being",

    # System prompt extraction
    r"(show|reveal|display|print|output)\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions?|message)",
    r"what\s+(is|are)\s+your\s+(system|original|initial)\s+(prompt|instructions?)",
    r"repeat\s+(your|the)\s+(system|initial)\s+(prompt|message)",

    # Delimiter injection
    r"```\s*(system|assistant|user)\s*```",
    r"\[INST\]|\[/INST\]",
    r"<<SYS>>|<</SYS>>",
    r"<\|im_start\|>|<\|im_end\|>",

    # Jailbreak attempts
    r"dan\s+mode|do\s+anything\s+now",
    r"developer\s+mode\s+(enabled|activated|on)",
    r"jailbreak|jail\s*break",
    r"bypass\s+(your\s+)?(safety|restrictions?|filters?|rules?)",

    # Encoding tricks
    r"base64\s*:\s*[A-Za-z0-9+/=]{20,}",
    r"hex\s*:\s*[0-9a-fA-F]{20,}",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


# =============================================================================
# Sensitive Data Patterns
# =============================================================================

SENSITIVE_PATTERNS = {
    "api_key": r"(sk-|api[_-]?key|apikey)[a-zA-Z0-9_-]{20,}",
    "password": r"password\s*[:=]\s*['\"]?[^\s'\"]{8,}",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "email_password": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s*[:=]\s*\S+",
    "private_key": r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "jwt": r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
}

COMPILED_SENSITIVE = {k: re.compile(v, re.IGNORECASE) for k, v in SENSITIVE_PATTERNS.items()}


# =============================================================================
# Detection Results
# =============================================================================

@dataclass
class SanitizationResult:
    """Result of sanitization check."""
    is_safe: bool
    original_text: str
    sanitized_text: str
    issues: list[str]
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class InjectionDetectionResult:
    """Result of prompt injection detection."""
    is_injection: bool
    confidence: float  # 0.0 to 1.0
    matched_patterns: list[str]
    explanation: str


# =============================================================================
# Detection Functions
# =============================================================================

def detect_prompt_injection(
    text: str,
    threshold: float = 0.5,
    raise_on_detection: bool = False,
) -> InjectionDetectionResult:
    """
    Detect potential prompt injection attempts.

    Args:
        text: Input text to analyze
        threshold: Confidence threshold for flagging (0.0-1.0)
        raise_on_detection: Raise PromptInjectionError if detected

    Returns:
        InjectionDetectionResult with detection details
    """
    if not text:
        return InjectionDetectionResult(
            is_injection=False,
            confidence=0.0,
            matched_patterns=[],
            explanation="Empty input"
        )

    matched_patterns = []

    # Check against compiled patterns
    for i, pattern in enumerate(COMPILED_PATTERNS):
        if pattern.search(text):
            matched_patterns.append(INJECTION_PATTERNS[i])

    # Calculate confidence based on matches
    if not matched_patterns:
        confidence = 0.0
    elif len(matched_patterns) == 1:
        confidence = 0.6
    elif len(matched_patterns) == 2:
        confidence = 0.8
    else:
        confidence = 0.95

    # Additional heuristics
    # Check for unusual character distributions
    special_char_ratio = len(re.findall(r'[<>\[\]{}|\\]', text)) / max(len(text), 1)
    if special_char_ratio > 0.1:
        confidence = min(1.0, confidence + 0.1)

    # Check for very long inputs (potential buffer overflow attempts)
    if len(text) > 10000:
        confidence = min(1.0, confidence + 0.05)

    is_injection = confidence >= threshold

    explanation = "No injection detected"
    if is_injection:
        explanation = f"Detected {len(matched_patterns)} suspicious patterns"

    result = InjectionDetectionResult(
        is_injection=is_injection,
        confidence=confidence,
        matched_patterns=matched_patterns,
        explanation=explanation,
    )

    if is_injection:
        logger.warning(
            "Potential prompt injection detected",
            metadata={
                "confidence": confidence,
                "pattern_count": len(matched_patterns),
                "text_length": len(text),
            }
        )

        if raise_on_detection:
            raise PromptInjectionError(
                message="Potential prompt injection detected",
                details={
                    "confidence": confidence,
                    "patterns_matched": len(matched_patterns),
                }
            )

    return result


def detect_sensitive_data(text: str) -> dict[str, list[str]]:
    """
    Detect sensitive data in text.

    Args:
        text: Text to scan

    Returns:
        Dictionary of detected sensitive data types and matches
    """
    detected = {}

    for data_type, pattern in COMPILED_SENSITIVE.items():
        matches = pattern.findall(text)
        if matches:
            # Mask the actual values for logging
            detected[data_type] = [f"[REDACTED {data_type.upper()}]"] * len(matches)

    if detected:
        logger.warning(
            "Sensitive data detected in text",
            metadata={"types": list(detected.keys())}
        )

    return detected


# =============================================================================
# Sanitization Functions
# =============================================================================

def sanitize_input(
    text: str,
    max_length: int = 50000,
    check_injection: bool = True,
    injection_threshold: float = 0.7,
    strip_sensitive: bool = True,
) -> SanitizationResult:
    """
    Sanitize user input before processing.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        check_injection: Whether to check for prompt injection
        injection_threshold: Threshold for injection detection
        strip_sensitive: Whether to redact sensitive data

    Returns:
        SanitizationResult with sanitized text and issues found
    """
    issues = []
    sanitized = text
    severity = "low"

    # Length check
    if len(text) > max_length:
        sanitized = text[:max_length]
        issues.append(f"Input truncated from {len(text)} to {max_length} characters")
        severity = "medium"

    # Null byte removal
    if "\x00" in sanitized:
        sanitized = sanitized.replace("\x00", "")
        issues.append("Removed null bytes")
        severity = "medium"

    # Control character removal (except newlines and tabs)
    control_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    if control_chars.search(sanitized):
        sanitized = control_chars.sub("", sanitized)
        issues.append("Removed control characters")

    # Prompt injection check
    if check_injection:
        injection_result = detect_prompt_injection(sanitized, threshold=injection_threshold)
        if injection_result.is_injection:
            issues.append(f"Prompt injection detected (confidence: {injection_result.confidence:.2f})")
            severity = "critical"

    # Sensitive data detection and redaction
    if strip_sensitive:
        for data_type, pattern in COMPILED_SENSITIVE.items():
            matches = pattern.findall(sanitized)
            if matches:
                sanitized = pattern.sub(f"[REDACTED_{data_type.upper()}]", sanitized)
                issues.append(f"Redacted {len(matches)} {data_type} instances")
                if severity != "critical":
                    severity = "high"

    is_safe = severity in ("low", "medium")

    if issues:
        logger.info(
            "Input sanitization applied",
            metadata={
                "issues_count": len(issues),
                "severity": severity,
                "original_length": len(text),
                "sanitized_length": len(sanitized),
            }
        )

    return SanitizationResult(
        is_safe=is_safe,
        original_text=text,
        sanitized_text=sanitized,
        issues=issues,
        severity=severity,
    )


def sanitize_output(
    text: str,
    strip_sensitive: bool = True,
    max_length: Optional[int] = None,
) -> SanitizationResult:
    """
    Sanitize LLM output before returning to user.

    Args:
        text: Output text to sanitize
        strip_sensitive: Whether to redact sensitive data
        max_length: Maximum output length (None for no limit)

    Returns:
        SanitizationResult with sanitized text
    """
    issues = []
    sanitized = text
    severity = "low"

    # Length check
    if max_length and len(text) > max_length:
        sanitized = text[:max_length]
        issues.append(f"Output truncated to {max_length} characters")
        severity = "medium"

    # Sensitive data redaction
    if strip_sensitive:
        for data_type, pattern in COMPILED_SENSITIVE.items():
            matches = pattern.findall(sanitized)
            if matches:
                sanitized = pattern.sub(f"[REDACTED]", sanitized)
                issues.append(f"Redacted {len(matches)} potential {data_type} in output")
                severity = "high"

    # Remove any system prompt leakage indicators
    system_leak_patterns = [
        r"my\s+system\s+prompt\s+is",
        r"my\s+instructions?\s+(are|is)",
        r"i\s+was\s+programmed\s+to",
    ]

    for pattern in system_leak_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            issues.append("Potential system prompt leakage detected")
            severity = "high" if severity != "critical" else severity

    is_safe = severity in ("low", "medium")

    return SanitizationResult(
        is_safe=is_safe,
        original_text=text,
        sanitized_text=sanitized,
        issues=issues,
        severity=severity,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def escape_for_prompt(text: str) -> str:
    """
    Escape text for safe inclusion in prompts.

    Wraps user content to prevent delimiter injection.
    """
    # Replace common delimiters
    escaped = text.replace("```", "'''")
    escaped = escaped.replace("---", "___")

    return escaped


def validate_json_output(text: str) -> bool:
    """
    Validate that LLM output is valid JSON.

    Useful for structured output validation.
    """
    import json
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
