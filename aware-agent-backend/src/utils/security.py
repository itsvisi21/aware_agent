from src.config.settings import Settings
import logging
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

class SecurityService:
    """Service for handling security-related operations."""
    
    def __init__(self, settings: Settings):
        """Initialize the security service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = settings.secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            plain_password: The password to verify
            hashed_password: The hashed password to verify against
            
        Returns:
            True if the password matches, False otherwise
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate a hash for a password.
        
        Args:
            password: The password to hash
            
        Returns:
            The hashed password
        """
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token.
        
        Args:
            data: The data to encode in the token
            expires_delta: Optional expiration time delta
            
        Returns:
            The encoded JWT token
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode a JWT token.
        
        Args:
            token: The token to decode
            
        Returns:
            The decoded token data
            
        Raises:
            jwt.InvalidTokenError: If the token is invalid
        """
        try:
            decoded_token = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return decoded_token
        except jwt.InvalidTokenError as e:
            logger.error(f"Error decoding token: {str(e)}")
            raise 