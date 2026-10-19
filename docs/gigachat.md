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

## 5. Как это встанет в probe (ещё не сделано)

- `probe/config.py`: поле `gigachat_credentials` и ветка в `require_key`.
- `probe/models.py`: провайдер `"gigachat"` и роль наблюдателя `GigaChat-2-Max`, `family="sber"`.
  Провайдер и модель входят в ключ кассеты, поэтому записи DeepSeek не смешаются с записями
  GigaChat.
- `probe/llm.py`: ветка `gigachat` в `_call_provider` через SDK. Строгой `json_schema` у GigaChat
  нет, поэтому делаем как с DeepSeek: схема идёт в системную инструкцию (`deepseek_system`,
  стоит переименовать в общее имя), ответ проверяем после получения. Невалидный JSON и
  выдуманные id считаем ошибками наблюдателя, а не чиним.
- Температуру выставить явно и записать в `design/02-validation.md`, как и точную версию модели
  из ответа (`served_model`).

## 6. Эксперимент

1. Прогнать наблюдателя GigaChat по 6 персонам в режиме `PROBE_MODE=live PROBE_RECORD=1`.
2. Сравнить с эталонной разметкой по тем же метрикам, что у DeepSeek: попадания, промахи, ложные
   срабатывания, выдуманные id, доля невалидного JSON.
3. Судью оставить прежним (DeepSeek). Тогда наблюдатель и судья из разных семейств, и снимается
   оговорка «одно семейство», которая сейчас стоит в отчётах.
4. Результат и ограничения записать в `design/02-validation.md`.

Отдельная ветка на потом: GigaChat Lightning, открытые веса на Hugging Face (`ai-sage`). Её можно
запускать локально на сервере из `demo/server.html`, то есть в закрытом контуре без отправки
диалогов наружу.
