from fastapi import APIRouter

from src.api.auth import router as auth_router
from src.api.user import router as user_router
from src.api.video import router as video_router

api_router = APIRouter()
api_router.include_router(auth_router, tags=["Auth"], prefix="/auth")
api_router.include_router(user_router, tags=["User"], prefix="/user")
api_router.include_router(video_router, tags=["Video"], prefix="/video")