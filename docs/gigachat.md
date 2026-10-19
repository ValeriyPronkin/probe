# GigaChat: как получить доступ и подключить к probe

Заметка от 2026-10-19. Цифры по тарифам взяты с developers.sber.ru в этот день. С 1 сентября
2026 года для новых клиентов оплата переносится на cloud.ru, поэтому перед регистрацией их стоит
проверить ещё раз.

## Зачем

Надо сравнить наблюдателя на GigaChat с тем же наблюдателем на DeepSeek на одинаковых
транскриптах и проверить согласие с эталонной разметкой. Это опыт работы с моделью, которую
выбирают на российском рынке, а сравнение даёт число, а не мнение.

## 1. Получить ключ (делает человек)

1. Войти на https://developers.sber.ru/studio через Сбер ID. Нужно **личное пространство**:
   Freemium есть только там.
2. Создать проект GigaChat API.
3. В настройках API нажать «Получить ключ». Скопировать **Authorization key**: это base64 от
   `client_id:client_secret`. Ключ показывают один раз.
4. Записать его в `probe/.env` (этот файл в `.gitignore`):

   ```
   GIGACHAT_CREDENTIALS=<Authorization key>
   GIGACHAT_SCOPE=GIGACHAT_API_PERS
   ```

Что даёт Freemium физлицу: 12 месяцев, 250 млн токенов Lite, 40 млн Pro и 25 млн Max. Один
поток, то есть запросы идут строго последовательно. Для 6 персон по 40–60 вызовов этого хватает
с большим запасом.

## 2. Сертификат Минцифры

Серверы GigaChat используют сертификат Российского доверенного корневого центра. На macOS его нет
по умолчанию, поэтому без него запрос падает с `SSL: CERTIFICATE_VERIFY_FAILED`.

```
mkdir -p ~/.certs
curl -o ~/.certs/russian_trusted_root_ca.pem \
  https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt
```

Отключать проверку (`verify_ssl_certs=False`) можно только для первого пробного запроса. В
приборе её не отключаем.

## 3. Проверка ключа из терминала

Токен доступа живёт 30 минут. Его получают из Authorization key:

```
set -a; source .env; set +a
TOKEN=$(curl -s --cacert ~/.certs/russian_trusted_root_ca.pem \
  -X POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -H 'Accept: application/json' \
  -H "RqUID: $(uuidgen)" \
  -H "Authorization: Basic $GIGACHAT_CREDENTIALS" \
  --data-urlencode "scope=$GIGACHAT_SCOPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -s --cacert ~/.certs/russian_trusted_root_ca.pem \
  https://gigachat.devices.sberbank.ru/api/v1/models \
  -H "Authorization: Bearer $TOKEN"
```

Если пришёл список моделей, ключ, сертификат и сеть работают. Если запрос завис или вернул 403,
сначала выключите VPN: API рассчитан на российские адреса.

Идентификаторы моделей: `GigaChat-2` (Lite), `GigaChat-2-Pro`, `GigaChat-2-Max`. Точный список
выдаёт `/models`.

## 4. Из Python

Официальный SDK: `uv add gigachat` (зависимость попадёт в `uv.lock`, как требует CLAUDE.md).

```python
from gigachat import GigaChat

with GigaChat(
    credentials=os.environ["GIGACHAT_CREDENTIALS"],
    scope="GIGACHAT_API_PERS",
    model="GigaChat-2-Max",
    ca_bundle_file=os.path.expanduser("~/.certs/russian_trusted_root_ca.pem"),
) as giga:
    print(giga.chat("Скажи «работает».").choices[0].message.content)
```

Токен SDK обновляет сам.

## 5. Как это сделано в probe

Проверено 2026-10-19:

- `probe/config.py` читает `GIGACHAT_CREDENTIALS` и `GIGACHAT_CA_BUNDLE`. По умолчанию
  `~/.certs/russian_trusted_root_ca.pem`, поэтому в `.env` достаточно одного ключа.
- `probe/models.py`: роль `GIGACHAT_OBSERVER` (`GigaChat-2-Max`, `family="sber"`). Наблюдатель
  по умолчанию остаётся DeepSeek, и все записанные сессии проигрываются как раньше. Провайдер
  входит в ключ кассеты, поэтому записи двух моделей не смешиваются.
- `probe/llm.py` ходит в GigaChat через SDK `gigachat`, токен SDK обновляет сам. Строгая
  `json_schema` в SDK помечена как бета: на схеме наблюдателя сервер ответил 500. Поэтому схема
  передаётся в системной инструкции (`schema_in_system`), как у DeepSeek, а ответ проверяется
  после получения.
- `temperature=0` сервер принимает.
- SDK теперь обращается к `api.giga.chat`, а не к `gigachat.devices.sberbank.ru`. Ключ тот же.
- Один ход наблюдателя занимает около двух минут (2600 токенов на входе), поток у физлица один.
  Шесть персон записывались около часа, без учёта перезапусков.
- `GigaChat-2-Max` — плавающий идентификатор: за один день сервер назвал версии 2.0.28.2 и
  2.0.30.01. Версию из каждого ответа пишем в кассету (`served_model`).

- Явно заданы `max_tokens=4096` и `profanity_check=False`. Ответ, обрезанный по лимиту
  (`finish_reason=length`) или заменённый отказом фильтра (`blacklist`), — ошибка, а не пустая
  разметка: сессия без балла лучше, чем балл без одного хода. Правила равных условий —
  `design/02-validation.md`, раздел «Равные условия для участников».

Запуск: `uv run scripts/compare_observers.py --record` записывает то, чего нет. Без `--record`
скрипт только проигрывает записи и печатает таблицу.

## 6. Эксперимент и результат (2026-10-19)

Наблюдатель GigaChat-2-Max прогнан на 6 персонах, судья прежний (DeepSeek). Подробности —
`design/02-validation.md`, раздел «Второй наблюдатель».

- Главный контраст воспроизводится: сильный участник впереди по всем конструктам.
- По заданиям расхождение с эталоном 2 из 48 (у DeepSeek 1 из 48).
- Каппа с эталоном 0.57 (у DeepSeek 0.58), между моделями тоже 0.57.
- Обе модели расходятся с эталоном на одних и тех же индикаторах (`kn.accept_silent`,
  `kn.scope`). Их определения стоит переписать первыми.
- Нарушения формата ответа: одно `strength: "medium"`, два голых `[]`. Прибор их считает и не
  падает.

Сбои API, которые встретились и теперь обрабатываются повтором с паузой (15, 60, 180 с):

| Сбой | Как выглядит |
| --- | --- |
| Ошибка сервера | `500 Internal Server Error` на часть одинаковых запросов |
| Зависание | нет ответа дольше 5 минут |
| Обрыв соединения | `SSL: UNEXPECTED_EOF_WHILE_READING` |
| Строгая схема | `response_format` с `json_schema` → 500 на большой схеме |

Отдельная ветка на потом: GigaChat Lightning, открытые веса на Hugging Face (`ai-sage`). Её можно
запускать локально на сервере из `demo/server.html`, то есть в закрытом контуре без отправки
диалогов наружу.
