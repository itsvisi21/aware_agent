"""Custom exceptions for the Aware Agent application."""

class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

class ProcessingError(Exception):
    """Exception raised for processing errors."""
    pass

class AgentTimeoutError(Exception):
    """Custom timeout error for agent operations."""
    pass


class AgentError(Exception):
    """Base class for agent-related errors."""
    pass


class AgentConfigurationError(AgentError):
    """Raised when there's an error in agent configuration."""
    pass


class AgentCommunicationError(AgentError):
    """Raised when there's an error in agent communication."""
    pass


class AgentProcessingError(AgentError):
    """Raised when there's an error in agent message processing."""
    pass


class ServiceError(Exception):
    """Base class for service-related errors."""
    pass


class AbstractionError(Exception):
    """Exception raised for errors in semantic abstraction."""
    pass
