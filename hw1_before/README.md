# hw_#1_mlOps

Для запуска api, запустить файл run_api.py

HSE MLOps 

Реализовать API (REST либо процедуры gRPC), которое умеет:
1. Обучать ML-модель с возможностью настройки гиперпараметров. При этом гиперпараметры для разных моделей могут быть разные. Минимальное количество классов моделей доступных для обучения == 2.
2. Возвращать список доступных для обучения классов моделей
3. Возвращать предсказание конкретной модели (как следствие, система должна уметь хранить несколько обученных моделей)
4. Обучать заново и удалять уже обученные модели

Оценка
- [4 балла] Работоспособность программы - то что ее можно запустить и она выполняет задачи, перечисленные в требованиях.
- [3 балла] Корректность и адекватность программы - корректная обработка ошибок, адекватный выбор структур классов, понятная документация (docstring-и адекатные здесь обязательны)
- [2 балла] Стиль кода - соблюдение стайлгайда. Буду проверять flake8 (не все ошибки на самом деле являются таковыми, но какие можно оставить – решать вам, насколько они критичны, списка нет)
- [1 балл] Swagger – Есть документация API (Swagger) с помощью flask- restx или аналога
- [2 балла] – Реализация и REST API, и gRPC

Дополнительные нюансы
- Принимать буду ссылкой на репозиторий (гитхаб, гитлаб, etc)
- Зависимости – фиксируйте. Lock файл poetry либо requirements
- Можно будет поправить или обжаловать, на что укажу, до конца дедлайна правок
- Сами дедлайны:
- Сдача ДЗ – до 23.10.2021 23:59
- Принятие правок (по ранее присланному дз) – до 30.10.2021 23:59
