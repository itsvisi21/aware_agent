from src.config.settings import Settings
import logging
from typing import Any, Dict, List, Optional, Union
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationService:
    """Service for validating data and inputs."""
    
    def __init__(self, settings: Settings):
        """Initialize the validation service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
    
    def validate_email(self, email: str) -> bool:
        """Validate an email address.
        
        Args:
            email: The email address to validate
            
        Returns:
            True if the email is valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_date_format(self, date_str: str, format: str = "%Y-%m-%d") -> bool:
        """Validate a date string against a format.
        
        Args:
            date_str: The date string to validate
            format: The expected date format
            
        Returns:
            True if the date string matches the format, False otherwise
        """
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that all required fields are present in the data.
        
        Args:
            data: The data to validate
            required_fields: List of required field names
            
        Returns:
            List of missing field names
        """
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    def validate_string_length(self, text: str, min_length: int = 0, max_length: Optional[int] = None) -> bool:
        """Validate that a string's length is within specified bounds.
        
        Args:
            text: The string to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length (if None, no maximum)
            
        Returns:
            True if the string length is valid, False otherwise
        """
        if len(text) < min_length:
            return False
        if max_length is not None and len(text) > max_length:
            return False
        return True
    
    def validate_numeric_range(self, value: Union[int, float], min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
        """Validate that a numeric value is within specified bounds.
        
        Args:
            value: The numeric value to validate
            min_value: Minimum allowed value (if None, no minimum)
            max_value: Maximum allowed value (if None, no maximum)
            
        Returns:
            True if the value is within range, False otherwise
        """
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True 