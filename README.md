# Agent traces collector

Локальный агент для AppWorld и GAIA с удалённой OpenAI-compatible моделью vLLM.
Сборщик измеряет всю работу агента над одним исходным запросом: все обращения
к модели, reasoning, действия и время инструментов. Обучение предиктора не входит
в проект. Инструкции запуска и контракт данных дополняются вместе с реализацией.

## Принятые ограничения эксперимента

- Предиктору доступен **только исходный текст** `instruction` / `Question`.
- Модель, промпт, инструменты, бюджеты и настройки фиксируются в эксперименте.
- Суммарные generated tokens включают reasoning и все промежуточные действия.
  Reasoning — часть completion tokens; складывать их повторно нельзя.
- Неизвестные счётчики остаются `null`. Пересчёт текста токенизатором — оценка.
- Неуспешные и оборванные попытки сохраняются с причиной остановки.
- В репозиторий не попадают ключи, данные бенчмарков, трассы и выгрузки.

## Официальные источники

Проверены 29 октября 2026 года:

- [AppWorld](https://github.com/StonyBrookNLP/appworld),
  [минимальный агент](https://github.com/StonyBrookNLP/appworld/blob/main/notebooks/minimal_agent.ipynb).
- [GAIA, карточка и Parquet-формат](https://huggingface.co/datasets/gaia-benchmark/GAIA).
- [Статья GAIA](https://arxiv.org/abs/2311.12983).
- [Официальная проверка ответов GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py).
- [vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

GAIA требует HF-доступ и принятие условий. Не публикуйте задания и производные
трассы в открытых репозиториях. Agent получает вложения и окружение; predictor —
только исходный текст, без имён файлов, уровня сложности и метаданных аннотатора.
