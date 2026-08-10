class AtelierAIError(Exception):
    """Base class for AI pipeline errors."""
    pass

class AIProviderError(AtelierAIError):
    """Raised for network timeouts, API downtime, or rate limits."""
    pass

class AIOutputError(AtelierAIError):
    """Raised when the LLM hallucinates malformed JSON."""
    pass