#!/bin/bash

# Проверка установки Python >= 3.12
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.12.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Ошибка: Требуется Python версии 3.12.0 или выше"
    echo "Текущая версия: $python_version"
    exit 1
fi

# Проверка установки uv
if ! command -v uv &> /dev/null; then
    echo "uv не установлен. Устанавливаем..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Добавляем uv в PATH для текущей сессии
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Настройка .env если его нет
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Файл .env создан из .env.example. Пожалуйста, отредактируйте его с вашими настройками."
        echo "Нажмите любую клавишу для продолжения после редактирования..."
        read -n 1
    else
        echo "Файл .env.example не найден. Создаем базовый .env..."
        echo "SECRET_KEY=" > .env
        echo "DATABASE_URL=sqlite+aiosqlite:///./aisee.db" >> .env
        echo "ADMIN_CREATE_SECRET=" >> .env
        echo "Пожалуйста, отредактируйте файл .env с вашими настройками."
        echo "Нажмите любую клавишу для продолжения после редактирования..."
        read -n 1
    fi
fi

# Установка зависимостей
echo "Устанавливаем зависимости..."
uv sync

# Применение миграций
echo "Применяем миграции базы данных..."
uv run alembic upgrade head

# Запуск проекта
echo "Запускаем проект..."
uv run -m src.main