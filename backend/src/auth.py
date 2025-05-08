from datetime import datetime, timedelta, timezone
import uuid
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

from src.core.db import get_db
from src.models import User

from src.core.config import secret_key

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

load_dotenv()

SECRET_KEY = secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def create_access_token(
    user_id: str, is_superuser: bool, is_paid: bool, subscription_type: str
) -> str:
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "user_id": str(user_id),
        "is_paid": is_paid,
        "is_superuser": is_superuser,
        "subscription_type": subscription_type,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id = uuid.UUID(payload.get("user_id"))

        if user_id is None:
            raise credentials_exception

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise credentials_exception

        return user

    except JWTError:
        raise credentials_exception


async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав",
        )
    return current_user
