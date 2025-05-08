import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from uvicorn import Config, Server

from src.core.db import create_all, engine
from src.api.main import api_router

# i am funny haha

# FastAPI initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all()
    try:
        yield
    finally:
        await engine.dispose()

app = FastAPI(
    lifespan=lifespan,
    root_path="/",
)
app.include_router(api_router)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def start_fastapi():
    print("Запуск FastAPI сервера...")
    config = Config(app=app, host="0.0.0.0", port=8000, log_level="info", reload=True)
    server = Server(config)
    return server

async def main():
    server = await start_fastapi()
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nForced stop")
    except Exception as e:
        print(f"Error: {e}")