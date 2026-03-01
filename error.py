class SLMAgentError(Exception):
    """Base exception class for all custom SLM Agent errors."""
    pass

class KillSwitchTriggeredError(SLMAgentError):
    """Raised instantly by the stream monitor when a forbidden string is detected."""
    def __init__(self, trigger_word: str, partial_output: str):
        self.trigger_word = trigger_word
        self.partial_output = partial_output
        super().__init__(f"Generation severed: Forbidden token '{trigger_word}' detected.")

class InsufficientDataError(SLMAgentError):
    """Raised when the model follows the system prompt fallback"""
    def __init__(self):
        super().__init__("The model reported insufficient context to complete the task.")

class TokenLimitExceededError(SLMAgentError):
    """Raised when the model hits the max_new_tokens ceiling"""
    def __init__(self, tokens_generated: int):
        self.tokens_generated = tokens_generated
        super().__init__(f"Generation capped: Model reached the {tokens_generated} token limit.")

class FormatViolationError(SLMAgentError):
    """Raised by the post-generation evaluator if the model's output contains conversational filler."""
    def __init__(self, message: str):
        super().__init__(f"Format violation: {message}")
