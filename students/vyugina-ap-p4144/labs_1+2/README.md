# itmo-mlops

## Лабораторная работа №1

* Блокнот с исследованиями данных и созданием модели: `research.ipynb`
* Сравнение фреймворков `Pandas` и `PySpark`: `pyspark_processing.ipynb`
* Локальная установка базы данных Qdrant:
    ```bash
    docker run -d -p 6333:6333 --name mlops_bd qdrant/qdrant
    ```
* Блокнот с созданием коллекции и загрузкой в неё обучающей выборки: `upload_bd.ipynb`

## Описание Airflow Pipeline

Airflow DAG `airbnb_pipeline` автоматизирует обработку данных Airbnb и их загрузку в векторную базу данных Qdrant. Pipeline выполняется ежедневно и состоит из трех основных этапов:

1. **Проверка новых данных** (`check_new_data`): 
   - Сканирует директорию данных на наличие новых CSV-файлов (по умолчанию `assets`)
   - Если файлы не найдены, задача пропускается

2. **Обработка данных** (`process_data`):
   - Считывает найденные CSV-файлы
   - Применяет предобработку данных с помощью класса `DataPreprocessor`
   - Подготавливает данные для загрузки в базу данных

3. **Сохранение в базу данных** (`save_to_qdrant`):
   - Загружает обработанные данные в векторную базу данных Qdrant
   - Обрабатывает и логирует возможные ошибки при загрузке

Для работы pipeline необходимо:
- Поместить CSV-файлы с новыми данными Airbnb в директорию, указанную в настройках (`settings.data_dir`)
- Убедиться, что база данных Qdrant запущена и доступна
- Запустить Airflow Scheduler и Webserver, как описано ниже

## Запуск сервисов
Для запуска контейнеров нужны переменные среды, прописанные в файле `config.env`, поэтому для любого сервиса команда следующая:
```bash
docker-compose --env-file config.env up -d <SERVICE_NAME>
```

## Запуск Airflow Pipeline через Docker

1. Запустите сервисы через Docker Compose:
    ```bash
    docker-compose --env-file config.env up -d
    ```

2. Откройте веб-интерфейс Airflow в браузере:
    ```
    http://localhost:8080
    ```

3. Войдите в систему, используя следующие учетные данные:
    - Username: admin
    - Password: admin

## Запуск Airflow Pipeline вручную

1. Установите зависимости:
    ```bash
    poetry install
    ```

2. Настройка переменной окружения PYTHONPATH для корректного импорта модулей:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

3. Настройка директории DAGs в Airflow (если DAG не отображается в интерфейсе):
    ```bash
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/mlops/dags
    ```
    
    Или создайте символическую ссылку:
    ```bash
    ln -sf $(pwd)/mlops/dags/airbnb_pipeline.py $(pwd)/airflow/dags/
    ```

4. Инициализация базы данных Airflow (если запускаете впервые):
    ```bash
    poetry run airflow db init
    ```

5. Создание пользователя Airflow (если запускаете впервые):
    ```bash
    poetry run airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    ```

6. Запуск Airflow Webserver:
    ```bash
    poetry run airflow webserver --port 8080
    ```

7. Откройте новый терминал и запустите Airflow Scheduler:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)  # Установите PYTHONPATH в новом терминале
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/mlops/dags  # Установите директорию DAGs в новом терминале
    poetry run airflow scheduler
    ```

8. Откройте веб-интерфейс Airflow в браузере:
    ```
    http://localhost:8080
    ```

9. Войдите в систему, используя созданные учетные данные (по умолчанию: admin/admin)

10. В веб-интерфейсе вы увидите DAG `airbnb_pipeline` в списке доступных DAG'ов
