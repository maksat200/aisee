import uuid
from typing import Optional, Dict, Any

from pydantic import BaseModel, EmailStr, field_validator, Field


class UserCreate(BaseModel):
    """Pydantic model for user creation data validation"""
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_paid: bool = False
    is_superuser: bool = False
    subscription_type: Optional[str] = None
    other_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @field_validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class UserUpdate(BaseModel):
    """Pydantic model for user update data validation"""
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_paid: Optional[bool] = None
    is_superuser: Optional[bool] = None
    subscription_type: Optional[str] = None
    other_data: Optional[Dict[str, Any]] = None
    
    @field_validator('password')
    def password_strength(cls, v):
        if v is not None and len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v
    
    def get_update_dict(self) -> Dict[str, Any]:
        """Returns a dictionary of fields that are not None"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class UserResponse(BaseModel):
    """Pydantic model for user response data"""
    id: uuid.UUID
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_paid: bool
    is_superuser: bool
    subscription_type: Optional[str]
    
    class Config:
        from_attributes =True


class UserFilter(BaseModel):
    """Pydantic model for user filtering options"""
    skip: int = 0
    limit: int = 100
    email: str = None
    first_name: str = None 
    last_name: str= None
    is_paid: Optional[bool] = None
    subscription_type: Optional[str] = None


class SubscriptionUpdate(BaseModel):
    """Pydantic model for subscription update data"""
    is_paid: bool
    subscription_type: Optional[str] = None

