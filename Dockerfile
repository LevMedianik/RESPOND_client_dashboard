# Базовый образ Python
FROM python:3.12-slim

# Установим зависимости для работы с pip и компиляции пакетов
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements.txt из корня проекта
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Указываем порт
ENV PORT=8000

# Запускаем FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]