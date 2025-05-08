from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth import create_access_token
from src.core.db import get_db
from src.models import User

router = APIRouter()


@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):

    # Ищем пользователя в базе
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not user.check_password(form_data.password):
        raise HTTPException(
            status_code=401,
            detail="Неверные учетные данные",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.subscription_type:
        subscription_type = "None"
    else:
        subscription_type = user.subscription_type

    # Создаем токен с информацией о правах пользователя
    access_token = create_access_token(
        user_id=user.id,
        is_superuser=user.is_superuser,
        is_paid=user.is_paid,
        subscription_type=subscription_type,
    )

    return {"access_token": access_token, "token_type": "bearer"}
