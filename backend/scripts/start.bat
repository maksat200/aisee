@echo off
setlocal

REM Проверка версии Python
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set python_version=%%I
set required_version=3.12.0

REM Извлекаем только первые 4 символа версии (major.minor)
for /f "tokens=1,2 delims=." %%a in ("%python_version%") do (
    set major=%%a
    set minor=%%b
)
for /f "tokens=1,2 delims=." %%a in ("%required_version%") do (
    set req_major=%%a
    set req_minor=%%b
)

if %major% LSS %req_major% (
    echo Ошибка: Требуется Python версии %required_version% или выше
    echo Текущая версия: %python_version%
    pause
    exit /b 1
)
if %major% EQU %req_major% if %minor% LSS %req_minor% (
    echo Ошибка: Требуется Python версии %required_version% или выше
    echo Текущая версия: %python_version%
    pause
    exit /b 1
)

REM Проверка установки uv
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo uv не установлен. Устанавливаем...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo Не удалось установить uv. Устанавливаем через pip...
        pip install uv
    )
)

REM Настройка .env если его нет
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo Файл .env создан из .env.example. Пожалуйста, отредактируйте его с вашими настройками.
        echo Нажмите любую клавишу для продолжения после редактирования...
        pause > nul
    ) else (
        echo Файл .env.example не найден. Создаем базовый .env...
        echo SECRET_KEY=> .env
        echo DATABASE_URL=sqlite+aiosqlite:///./aisee.db>> .env
        echo ADMIN_CREATE_SECRET=>> .env
        echo Пожалуйста, отредактируйте файл .env с вашими настройками.
        echo Нажмите любую клавишу для продолжения после редактирования...
        pause > nul
    )
)

REM Установка зависимостей
echo Устанавливаем зависимости...
call uv sync

REM Применение миграций
echo Применяем миграции базы данных...
call uv run alembic upgrade head

REM Запуск проекта
echo Запускаем проект...
call uv run -m src.main

pause