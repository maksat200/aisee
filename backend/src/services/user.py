import uuid
from typing import Optional, Dict, Any, List, Union

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from src.schemas import SubscriptionUpdate, UserCreate, UserFilter, UserResponse, UserUpdate
from src.models import User
from src.core.db import get_db_manager


class UserService:
    """
    Service for handling user operations with SQLAlchemy.
    """
    
    def __init__(self, db: AsyncSession = None):
        """
        Initialize UserService with an optional database session.
        If db is None, each method will create its own session.
        """
        self.db = db
    
    async def _get_db(self) -> AsyncSession:
        """
        Returns the current db session or creates a new one
        """
        if self.db:
            return self.db
        
        # Create a context manager for the db session
        # that will be cleaned up after the operation
        self._db_context = get_db_manager()
        self._db_session = await self._db_context.__aenter__()
        return self._db_session
    
    async def _cleanup_db(self):
        """
        Cleanup the db session if it was created by this service
        """
        if not self.db and hasattr(self, '_db_context'):
            await self._db_context.__aexit__(None, None, None)
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """
        Create a new user in the database
        """
        db = await self._get_db()
        try:
            # Check if user with this email already exists
            existing_user = await self.get_user_by_email(user_data.email)
            if existing_user:
                raise ValueError(f"User with email {user_data.email} already exists")
            
            # Create new user
            new_user = User(
                id=uuid.uuid4(),
                email=user_data.email,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                is_paid=user_data.is_paid,
                is_superuser=user_data.is_superuser,
                subscription_type=user_data.subscription_type,
                other_data=user_data.other_data
            )
            
            new_user.set_password(user_data.password)
            
            try:
                db.add(new_user)
                await db.commit()
                await db.refresh(new_user)
                return UserResponse.model_validate(new_user)
            except IntegrityError:
                await db.rollback()
                raise ValueError("Failed to create user due to integrity error")
        finally:
            await self._cleanup_db()
    
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """
        Get user by ID
        """
        db = await self._get_db()
        try:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        finally:
            await self._cleanup_db()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email
        """
        db = await self._get_db()
        try:
            result = await db.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()
        finally:
            await self._cleanup_db()
    
    async def update_user(
        self,
        user_id: uuid.UUID, 
        update_data: Union[UserUpdate, Dict[str, Any]]
    ) -> Optional[UserResponse]:
        """
        Update user information
        """
        if isinstance(update_data, dict):
            update_data = UserUpdate(**update_data)
        
        update_dict = update_data.get_update_dict()
        
        db = await self._get_db()
        try:
            # Check if user exists
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            # Handle password update separately if it's in the update data
            if 'password' in update_dict:
                password = update_dict.pop('password')
                user.set_password(password)
                db.add(user)
                
            # Apply remaining updates using SQLAlchemy update
            if update_dict:
                query = (
                    update(User)
                    .where(User.id == user_id)
                    .values(**update_dict)
                    .execution_options(synchronize_session="fetch")
                )
                await db.execute(query)
            
            await db.commit()
            
            # Get updated user
            updated_user = await self.get_user_by_id(user_id)
            return UserResponse.model_validate(updated_user) if updated_user else None
        finally:
            await self._cleanup_db()
    
    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """
        Delete a user by ID
        """
        db = await self._get_db()
        try:
            # Check if user exists
            user = await self.get_user_by_id(user_id)
            if not user:
                return False
            
            query = delete(User).where(User.id == user_id)
            await db.execute(query)
            await db.commit()
            return True
        finally:
            await self._cleanup_db()
    
    async def list_users(self, filter_params: UserFilter = None) -> List[UserResponse]:
        """
        List users with optional filtering
        """
        if filter_params is None:
            filter_params = UserFilter()
            
        db = await self._get_db()
        try:
            query = select(User)
            
            # Apply filters if provided
            if filter_params.is_paid is not None:
                query = query.where(User.is_paid == filter_params.is_paid)
            if filter_params.subscription_type:
                query = query.where(User.subscription_type == filter_params.subscription_type)
            if filter_params.first_name:
                query = query.where(User.first_name == filter_params.first_name)
            if filter_params.last_name:
                query = query.where(User.last_name == filter_params.last_name)
            if filter_params.email:
                query = query.where(User.email == filter_params.email)
                
            # Apply pagination
            query = query.offset(filter_params.skip).limit(filter_params.limit)
            
            result = await db.execute(query)
            users = result.scalars().all()
            return [UserResponse.model_validate(user) for user in users]
        finally:
            await self._cleanup_db()
    
    async def update_subscription(
        self,
        user_id: uuid.UUID,
        subscription_data: SubscriptionUpdate
    ) -> Optional[UserResponse]:
        """
        Update user subscription status
        """
        update_data = UserUpdate(
            is_paid=subscription_data.is_paid,
            subscription_type=subscription_data.subscription_type
        )
        return await self.update_user(user_id, update_data)
    
    async def count_users(self, is_paid: Optional[bool] = None) -> int:
        """
        Count total users with optional filtering
        """
        db = await self._get_db()
        try:
            query = select(User)
            if is_paid is not None:
                query = query.where(User.is_paid == is_paid)
                
            result = await db.execute(query)
            return len(result.scalars().all())
        finally:
            await self._cleanup_db()
