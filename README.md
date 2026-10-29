# Agent traces collector

Локальный агент для AppWorld и GAIA с удалённой OpenAI-compatible моделью vLLM.
Сборщик измеряет всю работу агента над одним исходным запросом: все обращения
к модели, reasoning, действия и время инструментов. Обучение предиктора не входит
в проект.

## Быстрый запуск

Linux/macOS, Python 3.12 и [uv](https://docs.astral.sh/uv/).
На этой машине окружение `.venv` и данные обоих бенчмарков уже подготовлены.
Для новой машины, из корня репозитория:

```bash
uv sync --frozen --python 3.12 --extra appworld --extra gaia --extra dev
source .venv/bin/activate
```

Точные версии находятся в `uv.lock`. Используется официальный PyPI-релиз
AppWorld **0.1.3**. Его ограничения требуют `httpx<0.28`, `click<8.2`, `ipython<9`;
они учтены в зависимостях. Не подменяйте его текущей веткой `main` внутри одного
эксперимента: там уже другой API и набор зависимостей.

Укажите параметры сервера и HF-доступ в окружении своего терминала:

```bash
export VLLM_BASE_URL='http://your-server:8000/v1'
export VLLM_MODEL='exact-served-model-id'
# При необходимости также задайте VLLM_API_KEY и HF_TOKEN.
```

В шаблонах Qwen3 `VLLM_MODEL` должен совпадать с ID из `GET /v1/models`, включая алиас
`--served-model-name`, если он задан. Ключи читаются из окружения; `.env` **не**
подгружается автоматически. Имена переменных есть в [.env.example](.env.example).
В проверенном профиле Gemma served ID уже задан явно.

Подготовка данных на новой машине:

```bash
trace-collector prepare-appworld --root data/appworld
trace-collector prepare-gaia --root data/gaia
```

`prepare-appworld` запускает официальный `appworld install` и загрузчик данных;
если папка данных уже существует, команда останавливается, чтобы не перезаписать её.
`appworld install` также разворачивает файлы пакета и кеш тестов AppWorld в `~/.appworld`.
GAIA скачивается из официального gated-репозитория с фиксацией commit SHA.
Сначала примите условия на HF и задайте `HF_TOKEN` с доступом к датасету.

Создайте локальные конфигурации и внесите точные сведения о сервере:

```bash
cp configs/gemma-appworld.toml configs/appworld.local.toml
cp configs/gemma-gaia.toml configs/gaia.local.toml
```

В `[model.server_metadata]` внесите известные checkpoint revision, quantization,
версию vLLM и reasoning parser. По одному `/models` нельзя проверить неизменность
весов и chat template. Выше выбран проверенный профиль Gemma 4. Для будущего
Qwen3 используйте шаблоны `configs/appworld.toml` и `configs/gaia.toml`;
они подготовлены, но пока не проверены на настоящем Qwen endpoint.
Семейство выбирается явно в `model.family`.

Проверка данных и сервера, затем по **3 задачи** каждого бенчмарка:

```bash
trace-collector doctor --config configs/appworld.local.toml --probe
trace-collector doctor --config configs/gaia.local.toml --probe
trace-collector run --config configs/appworld.local.toml --output runs/appworld-smoke --limit 3
trace-collector run --config configs/gaia.local.toml --output runs/gaia-smoke --limit 3
```

`--probe` делает отдельный запрос «2+2», расходует токены и печатает доступность
usage/reasoning; он не входит в трассы задач. Без `--probe` запросов генерации нет.
GAIA по согласованной конфигурации запускается **без поиска**, `search_provider="none"`.
Прямое чтение URL остаётся доступным. Это не конфигурация для сравнения с агентами,
имеющими полноценный поиск, браузер и мультимодальные инструменты.

Статистика и выгрузка:

```bash
trace-collector stats runs/appworld-smoke
trace-collector stats runs/gaia-smoke
trace-collector export runs/appworld-smoke --output exports/appworld-solved.jsonl
trace-collector export runs/gaia-smoke --output exports/gaia-solved.jsonl
```

По умолчанию выгружаются только завершённые **проверенно успешные** попытки
с полным `generated_tokens`. Неизвестный reasoning остаётся `null`.
`--require-reasoning` дополнительно требует полный счётчик reasoning.
`--include-unsolved` создаёт диагностическую выгрузку всех попыток, включая
ошибки, неизвестные значения и ограничения бюджета. Это данные о работе
**до остановки**, которые нельзя выдавать за стоимость успешного решения.
Для GAIA `test` ответы скрыты, поэтому `success=null`; для первоначального
сбора используйте `validation`.

## Длительные запуски и восстановление

```bash
# Продолжить ту же выборку и конфигурацию после остановки:
trace-collector run --config configs/gaia.local.toml --output runs/gaia-smoke --limit 3 --resume

# Полный сплит в новом эксперименте:
trace-collector run --config configs/gaia.local.toml --output runs/gaia-full --all

# Три повтора на каждое из трёх заданий:
trace-collector run --config configs/gaia.local.toml --output runs/gaia-repeats --limit 3 --repeats 3
```

`--task-id ID` можно указывать несколько раз; затем применяется `--limit` или `--all`.
Порядок задач — официальный порядок сплита. Повторы сохраняют тот же seed и
начинаются с пустой истории и новым окружением. Для оценки вариативности
генерации с разными seed используйте отдельные фиксированные конфигурации.

Завершённые попытки при `--resume` пропускаются, даже если они неуспешны.
`--resume --retry-failed` запускает их заново, сохраняя прежние попытки.
Прерванная попытка помечается `interrupted`; продолжение начинает её заново,
без восстановления истории и без изменения старой трассы.

Нельзя продолжить каталог эксперимента с изменённым кодом, зависимостями,
конфигурацией, данными выбранных задач или числом повторов. Изменение
содержательных полей ответа `/models` тоже требует нового каталога; динамические
`created` и permission IDs игнорируются. Одновременная запись в один каталог
блокируется. Запуски последовательные: конкурентность не смешивается с
измерениями времени и AppWorld-состоянием.

## Что реализовано

- Собственный текстовый агент через Chat Completions с отдельными
  [профилями семейств](configs/README.md): JSON-действия для Qwen3,
  нативные действия и финальный текст для Gemma 4. Серверный function-calling
  parser ему не требуется. Нет скрытых вызовов judge-модели.
- AppWorld: штатный `world.execute`, `task_completed` и state-based `world.evaluate`.
  Задания стартуют с чистого состояния; агент не получает ground truth или сложность.
- GAIA: официальный Parquet, копирование вложений в отдельный workspace,
  Python с сохраняемыми переменными, чтение PDF/DOCX/XLSX/текста/списка ZIP,
  HTTP-страниц и PDF. Нормализация ответов повторяет официальный scorer.
- Необязательный Brave Search: задайте `search_provider="brave"` и
  `BRAVE_API_KEY` в **новом** эксперименте. Сейчас он отключён.
- Сырые запросы/ответы модели, reasoning, наблюдения инструментов, ошибки,
  ретраи и временные метки сохраняются в JSONL с `flush`/`fsync` после событий.
- Отдельные процессы для окружений, принудительные таймауты инструментов,
  реальные часы вне AppWorld, лимиты шагов/генерации/времени, атомарные итоги
  и восстановление незавершённых трасс.

Python инструмента GAIA выполняет локальный код и не является защитной
песочницей. Для бенчмарков используйте отдельную машину/контейнер без личных
данных. В этой конфигурации нет встроенных vision, OCR, аудиотранскрипции,
обработки видео и интерактивного браузера; часть заданий GAIA решена не будет.
Такие попытки сохраняются, а успешными метками не становятся.

## Учёт токенов

Для проверенного сервера Gemma 4 12B предусмотрены
`configs/gemma-appworld.toml` и `configs/gemma-gaia.toml`:

```bash
export VLLM_BASE_URL='http://everest.nsu.ru:9022/v1'
trace-collector doctor --config configs/gemma-appworld.toml --probe
trace-collector run --config configs/gemma-appworld.toml --output runs/gemma-appworld --limit 3
trace-collector run --config configs/gemma-gaia.toml --output runs/gemma-gaia --limit 3
```

В этих конфигурациях thinking включён явно. Этот сервер не возвращает reasoning
breakdown, поэтому используется точный счётчик по **реальным output token IDs**.
Маркеры сначала сверяются с серверным `/tokenize`; при несовпадении запуск
останавливается. `skip_special_tokens=false` сохраняет границы каналов,
сборщик отделяет reasoning перед разбором JSON-действия.

Токены строго внутри `<|channel>thought\n ... <channel|>` — `reasoning_tokens`,
сами сгенерированные маркеры — `reasoning_overhead_tokens`, остальные —
`response_tokens` (включая EOS, если сервер учитывает его в output IDs).
`non_reasoning_tokens = response_tokens + reasoning_overhead_tokens`.
Источник счётчика — `output_token_ids`; текст не ретокенизируется.
Если токенов/границ не хватает для точного счёта, reasoning остаётся `null`.
Для обычного режима без token ID accounting действуют правила server usage ниже.

Параметры Gemma соответствуют официальному
[generation_config](https://huggingface.co/google/gemma-4-12B-it/blob/main/generation_config.json):
`temperature=1.0`, `top_p=0.95`, `top_k=64`. Внутри инструментального хода
Gemma получает прежний reasoning через `assistant.reasoning` и результаты через
`role=tool`; Qwen3 использует историю без reasoning. Эти различия входят в
фиксированную конфигурацию и SHA кода эксперимента.

Для каждого запроса пользователя суммируются **все** генерации агента:

```text
generated_tokens = сумма usage.completion_tokens по всем вызовам модели
reasoning_tokens = сумма usage.completion_tokens_details.reasoning_tokens
non_reasoning_tokens = generated_tokens - reasoning_tokens
```

При двух вызовах по 100 completion tokens, включая по 60 reasoning,
результат — **200 generated, 120 reasoning, 80 non-reasoning**. Не 320.
`non_reasoning_tokens` включает JSON-действия, код и промежуточные ответы;
это не только длина финального ответа «4». Формат работы фиксирован для эксперимента.
Ответ инструмента сам по себе не является генерацией модели. Если он вошёл
в следующий промпт, сервер учитывает его в `prompt_tokens` следующего вызова.

Поддерживаются `message.reasoning`, старое `reasoning_content`, а также сырые
теги `<think>`. Разделение текста не доказывает точное количество сгенерированных
токенов: специальные токены и границы токенизации могут отличаться.
Поэтому без серверного breakdown или явно настроенного подсчёта реальных
output token IDs `reasoning_tokens=null`. Противоречивый
серверный ноль при непустом reasoning также отвергается.

Если нужно, установите extra `tokenizer` и задайте **точный** токенизатор с revision:

```bash
uv sync --frozen --extra appworld --extra gaia --extra dev --extra tokenizer
```

В `[accounting]` укажите `tokenizer` и `tokenizer_revision` из checkpoint сервера.
Пересчёт видимых текстов сохраняется как `*_text_tokens_estimate` в выгрузке и
`text_token_estimates` в трассах/итогах. Он никогда не подменяет server usage.
`reasoning_mode="disabled"` разрешено только при действительно выключенном
thinking; тогда reasoning явно принимается равным нулю, источник отмечается.

После HTTP timeout или неуспешного запроса сервер мог потратить ненаблюдаемые
токены. При этом полный итог равен `null`, известная часть остаётся в
`observed_generated_tokens`. По умолчанию ретраев нет; при включении каждый
ретрай записывается, и неизвестный расход не исчезает из итогов.

## Формат и проверки

[Контракт данных, время и ограничения](docs/data-contract.md).
[Параметры vLLM и разбор типичных проблем](docs/operations.md).

```bash
ruff check src tests
pytest -q
# Дополнительно с настоящими установленными данными AppWorld:
APPWORLD_TEST_ROOT="$PWD/data/appworld" pytest -q
```

Тесты используют временные синтетические задания и ответы HTTP-заглушки,
а интеграционная проверка AppWorld — настоящий engine и evaluator.
Это проверка сборщика, а не измерение качества или стоимости настоящей модели.
Gemma проверена на реальных прогонах по 3 задачи: AppWorld — 1/3 успешных,
GAIA без поиска — 0/3. Все счётчики полные; незавершённые попытки не выдаются
за успешные метки. [Подробный отчёт](docs/smoke-results.md).
Трассы и выгрузки сохраняются локально в `runs/` и `exports/`.

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
- [Thinking в Gemma 4](https://ai.google.dev/gemma/docs/capabilities/thinking),
  [маркеры Gemma в vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/parser/gemma4.py).

GAIA требует HF-доступ и принятие условий. Не публикуйте задания и производные
трассы в открытых репозиториях. Agent получает вложения и окружение; predictor —
только исходный текст, без имён файлов, уровня сложности и метаданных аннотатора.
