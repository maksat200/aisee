import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth import get_current_user, get_current_superuser
from src.core.db import get_db
from src.models import User
from src.schemas import (
    SubscriptionUpdate,
    UserCreate,
    UserFilter,
    UserResponse,
    UserUpdate,
)
from src.services.user import UserService
from src.core.config import admin_create_secret

router = APIRouter()


# Helper to get UserService with DB session
def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db=db)


# --- Public Endpoint for Registration ---


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service),
):
    """Public endpoint to register a new user"""
    # Override certain fields for security reasons
    user_data.is_superuser = False  # Prevent self-assignment of admin privileges
    user_data.is_paid = False

    try:
        return await user_service.create_user(user_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/admin/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register_admin(
    secret_key: str,
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service),
):
    """Public endpoint to register a new user"""
    
    if secret_key != admin_create_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Не сегодня шалунишка!",
        )
        
    user_data.is_superuser = True  # Prevent self-assignment of admin privileges
    user_data.is_paid = True
    
    # what is the size of your
    dict()
    # in python

    try:
        return await user_service.create_user(user_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# --- Regular User Endpoints ---


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
):
    """Get information about the current authenticated user"""
    user = await user_service.get_user_by_id(current_user.id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
):
    """Update current user's information"""
    # Prevent users from changing their own privileges
    if hasattr(update_data, "is_superuser"):
        update_data.is_superuser = None
    if hasattr(update_data, "is_paid"):
        update_data.is_paid = None
    if hasattr(update_data, "subscription_type"):
        update_data.subscription_type = None

    updated_user = await user_service.update_user(current_user.id, update_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Failed to update user",
        )
    return updated_user


# --- Admin Endpoints ---


@router.post("/admin", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    _: User = Depends(get_current_superuser),  # Only admins can create users
    user_service: UserService = Depends(get_user_service),
):
    """Create a new user (admin only)"""
    try:
        return await user_service.create_user(user_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/admin", response_model=List[UserResponse])
async def list_users(
    filter_params: UserFilter = Depends(),
    _: User = Depends(get_current_superuser),  # Only admins can list users
    user_service: UserService = Depends(get_user_service),
):
    """List all users with optional filtering (admin only)"""
    return await user_service.list_users(filter_params)


@router.get("/admin/count", response_model=int)
async def count_users(
    is_paid: Optional[bool] = None,
    _: User = Depends(get_current_superuser),  # Only admins can count users
    user_service: UserService = Depends(get_user_service),
):
    """Count users with optional filtering by payment status (admin only)"""
    return await user_service.count_users(is_paid)


@router.get("/admin/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    _: User = Depends(get_current_superuser),  # Only admins can get other users
    user_service: UserService = Depends(get_user_service),
):
    """Get user by ID (admin only)"""
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)


@router.patch("/admin/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    update_data: UserUpdate,
    _: User = Depends(get_current_superuser),  # Only admins can update other users
    user_service: UserService = Depends(get_user_service),
):
    """Update user's information (admin only)"""
    updated_user = await user_service.update_user(user_id, update_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return updated_user


@router.delete("/admin/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    _: User = Depends(get_current_superuser),  # Only admins can delete other users
    user_service: UserService = Depends(get_user_service),
):
    """Delete user (admin only)"""
    success = await user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.patch("/admin/{user_id}/subscription", response_model=UserResponse)
async def update_user_subscription(
    user_id: uuid.UUID,
    subscription_data: SubscriptionUpdate,
    _: User = Depends(
        get_current_superuser
    ),  # Only admins can update other users' subscriptions
    user_service: UserService = Depends(get_user_service),
):
    """Update user's subscription status (admin only)"""
    updated_user = await user_service.update_subscription(user_id, subscription_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return updated_user
