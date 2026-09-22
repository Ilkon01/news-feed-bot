"""
Персональный новостной бот (бесплатная версия на Google Gemini).

Как работает:
  ШАГ 1 — берём только свежие новости (по времени публикации)
  ШАГ 2 — отсев по ключевым словам (бесплатно, без ИИ)
  ШАГ 3 — дедупликация: одинаковые новости из разных источников
          склеиваются в один сюжет
  ШАГ 4 — для каждого сюжета ищем полный текст статьи: от самого
          надёжного источника, у которого текст открыт
  ШАГ 5 — Gemini: проверка по интересам + заголовок, текст
          и подробный конспект на русском
  ШАГ 6 — отправка в Telegram

Плюс раз в день утром — короткий прогноз погоды.

Настройки — только в config.yaml.
"""

import os
import re
import html
import time
import json
import calendar
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

import requests
import feedparser
import yaml
from bs4 import BeautifulSoup

import warnings
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
STATE_PATH = os.path.join(os.path.dirname(__file__), "state.json")
MAX_SEEN_ITEMS = 8000

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
# Куда слать: личный chat_id или канал (@имя_канала либо -100...)
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Модели пробуются по порядку: если первая перегружена, сразу берётся вторая.
GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]


def gemini_url(model):
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "send": {"type": "BOOLEAN"},
        "topic": {"type": "STRING"},
        "country": {"type": "STRING"},
        "importance": {"type": "STRING"},
        "title_ru": {"type": "STRING"},
        "lead": {"type": "STRING"},
        "detail": {"type": "STRING"},
    },
    "required": [
        "send", "topic", "country", "importance", "title_ru", "lead", "detail"
    ],
}

# Служебные слова, которые не участвуют в сравнении заголовков
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "will", "would", "can", "could", "has", "have", "had", "it", "its", "his",
    "her", "their", "this", "that", "these", "those", "after", "over", "into",
    "says", "said", "new", "amid", "up", "out", "no", "not", "who", "what",
    "и", "в", "во", "на", "с", "со", "по", "за", "из", "от", "до", "к", "у",
    "о", "об", "для", "что", "как", "это", "не", "но", "а", "же", "бы", "ли",
    "он", "она", "они", "его", "её", "их", "был", "была", "были", "будет",
    "также", "более", "свою", "этом", "который", "которые", "после", "может",
    "года", "году", "результате", "заявил", "сообщил", "стало", "стали",
}


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}
    state.setdefault("seen", [])
    state.setdefault("recent", [])     # оригинальные заголовки
    state.setdefault("recent_ru", [])  # русские заголовки после перевода
    return state


def save_state(state, memory_hours):
    state["seen"] = state["seen"][-MAX_SEEN_ITEMS:]
    cutoff = time.time() - memory_hours * 3600
    state["recent"] = [r for r in state["recent"] if r.get("ts", 0) > cutoff]
    state["recent_ru"] = [r for r in state["recent_ru"] if r.get("ts", 0) > cutoff]
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def check_telegram():
    """Проверяет связку токен + chat_id ДО начала работы и печатает
    понятное объяснение, если что-то не так."""
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    try:
        me = requests.get(f"{base}/getMe", timeout=15).json()
    except Exception as e:
        print(f"[ПРОВЕРКА] нет связи с Telegram: {e}")
        return True  # сетевой сбой не повод останавливать запуск

    if not me.get("ok"):
        print("[ПРОВЕРКА] ТОКЕН НЕВЕРНЫЙ ИЛИ ОТОЗВАН.")
        print("           Возьмите новый токен в @BotFather (/mybots -> API Token)")
        print("           и впишите его в секрет TELEGRAM_BOT_TOKEN.")
        return False

    # Репозиторий публичный, логи видны всем — поэтому имя бота
    # и название канала в лог не пишем.
    print("[ПРОВЕРКА] токен рабочий")

    cid = TELEGRAM_CHAT_ID.strip()
    if cid != TELEGRAM_CHAT_ID:
        print("[ПРОВЕРКА] в chat_id были лишние пробелы или перенос строки —"
              " поправьте секрет TELEGRAM_CHAT_ID")

    try:
        chat = requests.get(
            f"{base}/getChat", params={"chat_id": cid}, timeout=15
        ).json()
    except Exception as e:
        print(f"[ПРОВЕРКА] не удалось проверить чат: {e}")
        return True

    if chat.get("ok"):
        print(f"[ПРОВЕРКА] канал найден (тип: {chat['result'].get('type')})")
        return True

    print(f"[ПРОВЕРКА] ЧАТ НЕ НАЙДЕН: {chat.get('description', '')}")
    print(f"           длина chat_id: {len(cid)} знаков")
    if not cid.startswith("-100"):
        print("           ВНИМАНИЕ: идентификатор канала должен начинаться")
        print("           с -100. Сейчас он начинается иначе — скорее всего")
        print("           это и есть причина. Возьмите число из блока")
        print("           channel_post в ответе getUpdates.")
    else:
        print("           Проверьте, что бот добавлен")
        print("           администратором канала с правом отправки сообщений.")
    return False


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        if not resp.ok:
            print("Ошибка отправки в Telegram:", resp.text[:200])
            return False
        return True
    except Exception as e:
        print("Исключение при отправке в Telegram:", e)
        return False


def entry_age_hours(entry):
    tm = entry.get("published_parsed") or entry.get("updated_parsed")
    if not tm:
        return None
    published = datetime.fromtimestamp(calendar.timegm(tm), tz=timezone.utc)
    return (datetime.now(timezone.utc) - published).total_seconds() / 3600


def clean_html(raw):
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(" ").strip()


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def fetch_rss(src):
    """Скачиваем ленту сами, с браузерным заголовком: часть сайтов
    (в том числе Google News) отклоняет запросы от библиотек.
    Возвращает (записи, строка для лога)."""
    name, url = src["name"], src["url"]
    try:
        resp = requests.get(
            url,
            timeout=25,
            headers={
                "User-Agent": BROWSER_UA,
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            },
        )
        if resp.status_code != 200:
            return [], f"[{name}] сервер ответил {resp.status_code}"
        feed = feedparser.parse(resp.content)
    except Exception as e:
        return [], f"[{name}] ошибка загрузки: {type(e).__name__}"

    items = []
    for entry in feed.entries:
        link = entry.get("link", "")
        if not link:
            continue

        title = entry.get("title", "")
        real_source = name
        aggregate = src.get("aggregate", False)
        # У сводных лент Google News настоящий издатель лежит в поле source,
        # а в конце заголовка идёт хвост вида « - Reuters». Достаём и то, и то.
        if aggregate:
            src_field = entry.get("source")
            if isinstance(src_field, dict):
                publisher = (src_field.get("title") or "").strip()
                if publisher:
                    real_source = publisher
            if real_source == name:
                m = re.search(r"\s+-\s+([^-]{2,40})$", title)
                if m:
                    real_source = m.group(1).strip()
            real_source = clean_publisher(real_source) or name
        # убираем хвост с издателем из самого заголовка
        title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title).strip()

        items.append(
            {
                "id": link,
                "source": real_source,
                "feed": name,
                "country": src.get("country", ""),
                "trust": src.get("trust", 5),
                "kz": bool(src.get("kz", False)),
                "skip_keywords": bool(src.get("skip_keywords", False)),
                "title": title,
                "text": clean_html(entry.get("summary", "")),
                "link": link,
                "age_hours": entry_age_hours(entry),
            }
        )
    if not items:
        return items, f"[{name}] ВНИМАНИЕ: ноль записей (проверьте источник)"
    return items, f"[{name}] записей: {len(items)}"


def clean_publisher(name):
    """Приводит имя издателя из Google News к человеческому виду:
    «ABC News - Breaking News, Latest News and Videos» -> «ABC News»,
    «Problem Solvers Caucus (.gov)» -> «Problem Solvers Caucus»."""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)      # хвост в скобках
    name = re.split(r"\s+[-–|:]\s+", name)[0]          # всё после тире
    name = re.sub(r"\s*\.(com|net|org|ru|kz|kg|uz)\s*$", "", name, flags=re.I)
    return name.strip()[:40]


def domain_of(url):
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1).lower().replace("www.", "") if m else ""


def is_blocked(item, blocked):
    """Отсев нежелательных изданий по домену или названию."""
    dom = domain_of(item.get("link", ""))
    src_name = (item.get("source") or "").lower()
    for b in blocked:
        b = b.lower().strip()
        if not b:
            continue
        if b.startswith("."):          # зона целиком, например .ru
            if dom.endswith(b):
                return True
        elif b in dom or b in src_name:
            return True
    return False


def decode_google_news(url):
    """Ссылки из лент Google News ведут не на статью, а на страницу-
    переадресацию Google. Скачать с неё текст нельзя — поэтому раньше
    у Reuters, AP и Bloomberg бот видел только заголовок.
    Здесь по той же схеме, что использует сам сайт Google News,
    получаем настоящий адрес статьи. При любой неудаче — пустая строка,
    и бот работает как раньше."""
    m = re.search(r"news\.google\.com/(?:rss/)?(?:articles|read)/([^?/#]+)",
                  url or "")
    if not m:
        return ""
    art_id = m.group(1)
    try:
        page = requests.get(
            f"https://news.google.com/rss/articles/{art_id}",
            timeout=12,
            headers={"User-Agent": BROWSER_UA},
        )
        if page.status_code != 200:
            return ""
        soup = BeautifulSoup(page.text, "html.parser")
        div = soup.select_one("c-wiz > div[jscontroller]")
        if not div:
            return ""
        sig, ts = div.get("data-n-a-sg"), div.get("data-n-a-ts")
        if not sig or not ts:
            return ""
        req = [
            "Fbv4je",
            '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,'
            'null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,'
            f'null,0],"{art_id}",{ts},"{sig}"]',
        ]
        resp = requests.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "User-Agent": BROWSER_UA,
            },
            data="f.req=" + quote(json.dumps([[req]])),
            timeout=12,
        )
        if resp.status_code != 200:
            return ""
        chunk = json.loads(resp.text.split("\n\n")[1])
        real = json.loads(chunk[0][2])[1]
        return real if isinstance(real, str) and real.startswith("http") else ""
    except Exception:
        return ""


def fetch_article_text(url, max_chars=8000):
    """Заходит на страницу статьи и достаёт её текст.
    При любой неудаче возвращает пустую строку — тогда используется
    то, что было в ленте."""
    try:
        resp = requests.get(
            url,
            timeout=12,
            allow_redirects=True,
            headers={
                "User-Agent": BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            },
        )
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")

        # выкидываем всё, что не является текстом статьи
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "figure"]):
            tag.decompose()

        # берём абзацы разумной длины — так отсеиваются подписи и меню
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 60
        ]
        text = " ".join(paragraphs)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def pick_best_text(group, min_chars, blocked, max_attempts=3):
    """Выбирает из сюжета источник с открытым полным текстом.

    group — все издания, написавшие об этом событии, по убыванию доверия.
    Идём сверху вниз: если у самого надёжного текст доступен — берём его.
    Если закрыт (платная подписка, защита от ботов) — пробуем следующего.
    Возвращает (выбранный источник, сколько текстов догружено)."""
    best, best_len, fetched = None, -1, 0
    for cand in group[:max_attempts]:
        if "news.google.com" in cand["link"]:
            real = decode_google_news(cand["link"])
            if real:
                cand["link"] = real
                # настоящий адрес мог оказаться у заблокированного издания
                if is_blocked(cand, blocked):
                    continue
        text = cand.get("text") or ""
        if len(text) < min_chars:
            full = fetch_article_text(cand["link"])
            if len(full) > len(text):
                cand["text"] = full
                text = full
                fetched += 1
        if len(text) > best_len:
            best, best_len = cand, len(text)
        if best_len >= min_chars:
            break
    return best, fetched


def is_fresh(item, max_age_hours):
    if item["age_hours"] is None:
        return True
    return item["age_hours"] <= max_age_hours


def build_keyword_patterns(keywords):
    patterns = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        if kw.endswith("*"):
            patterns.append(re.compile(r"\b" + re.escape(kw[:-1]), re.IGNORECASE))
        else:
            patterns.append(re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
    return patterns


def passes_keywords(item, patterns):
    # тематические ленты (наука, здоровье, стартапы) идут к ИИ целиком
    if item.get("skip_keywords"):
        return True
    haystack = f"{item['title']} {item['text']}"
    return any(p.search(haystack) for p in patterns)


# ---------- Дедупликация ----------

def stem(word):
    """Грубая обрезка до основы: «израильской» и «израильская» → «израил».
    Нужна, чтобы русские окончания не мешали сравнению."""
    return word[:6] if len(word) > 6 else word


def title_tokens(title):
    """Значимые слова заголовка для сравнения."""
    # убираем хвост вроде " - Reuters", который добавляет Google News
    title = re.sub(r"\s+[-–|]\s+[^-–|]{2,30}$", "", title)
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", title.lower())
    return {stem(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def similarity(tokens_a, tokens_b):
    """Доля общих слов (мера Жаккара)."""
    if not tokens_a or not tokens_b:
        return 0.0
    common = len(tokens_a & tokens_b)
    return common / min(len(tokens_a), len(tokens_b))


def is_repeat_ru(data, recent_ru, threshold):
    """Сравнивает русский заголовок+пересказ с уже отправленными."""
    tokens = title_tokens(
        f"{data.get('title_ru', '')} {data.get('lead', '')[:250]}"
    )
    if not tokens:
        return False, set()
    for r in recent_ru:
        if similarity(tokens, set(r["tokens"])) >= threshold:
            return True, tokens
    return False, tokens


def deduplicate(items, threshold, recent):
    """Оставляет по одной новости на сюжет — от источника с высшим доверием.
    Также убирает то, что уже отправлялось в прошлых запусках."""
    for item in items:
        item["tokens"] = title_tokens(item["title"])

    # 1) убрать совпадающее с уже отправленным ранее
    recent_sets = [set(r["tokens"]) for r in recent]
    not_repeated, repeated = [], []
    for item in items:
        if any(similarity(item["tokens"], rs) >= threshold for rs in recent_sets):
            repeated.append(item)
        else:
            not_repeated.append(item)

    # 2) сгруппировать похожие между собой, оставить сильнейший источник
    groups = []  # список списков
    for item in sorted(not_repeated, key=lambda i: -i["trust"]):
        placed = False
        for group in groups:
            if similarity(item["tokens"], group[0]["tokens"]) >= threshold:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    winners = []
    for g in groups:
        # весь сюжет целиком, по убыванию доверия: если у первого
        # текст статьи окажется закрыт, возьмём у следующего
        g[0]["group"] = g
        winners.append(g[0])
    losers = [i for g in groups for i in g[1:]]
    return winners, losers + repeated


# ---------- Gemini ----------

LEVELS = [
    ("очень_интересно", "ОЧЕНЬ ИНТЕРЕСНО"),
    ("интересно", "ИНТЕРЕСНО"),
    ("средний_интерес", "СРЕДНИЙ ИНТЕРЕС"),
    ("слабый_интерес", "СЛАБЫЙ ИНТЕРЕС"),
]


def topics_text(config):
    """Разделы интересов с уровнями — в виде, удобном для модели."""
    lines = []
    for section, body in (config.get("topics") or {}).items():
        lines.append(f"■ {section}")
        for key, label in LEVELS:
            items = (body or {}).get(key) or []
            if items:
                lines.append(f"  {label}: " + "; ".join(items))
    return "\n".join(lines)


def bullets(items):
    return "\n".join(f"- {x}" for x in items or [])


def build_prompt(item, config):
    never = bullets(config.get("never_send"))
    rules_text = bullets(config.get("editorial_rules"))
    writing_text = bullets(config.get("writing_rules"))
    profile = bullets(config.get("reader_profile"))
    breaking = (
        "Также присылай крупные срочные мировые новости, даже если они "
        "не попадают ни под одну тему."
        if config.get("breaking_news")
        else ""
    )

    detail_len = config.get("detail_length", "6–12 предложений")
    lead_max = config.get("lead_max_chars", 125)
    title_max = config.get("title_max_chars", 45)

    dom = domain_of(item.get("link", ""))
    strict_dom = any(
        dom.endswith(d.lower().strip())
        for d in config.get("strict_domains", [])
        if d.strip()
    )

    strict = ""
    if item["source"] in config.get("strict_sources", []) or strict_dom:
        strict = (
            "\nОСОБОЕ УКАЗАНИЕ ПО ЭТОМУ ИСТОЧНИКУ: он склонен к тенденциозной "
            "подаче и продвижению определённой позиции. Будь к нему намного "
            "строже обычного. Присылай ТОЛЬКО сообщения о конкретных "
            "свершившихся фактах и событиях. Отклоняй (send: false) аналитику, "
            "колонки, мнения, прогнозы, а также материалы с оценочной или "
            "агитационной риторикой.\n"
        )

    return f"""Ты — персональный новостной редактор. Решаешь, нужна ли эта
новость читателю, и если да — пишешь её для Telegram-канала.

О ЧИТАТЕЛЕ:
{profile}

НОВОСТЬ:
Источник: {item['source']}
Заголовок: {item['title']}
Текст статьи: {item['text'][:7000]}

ИНТЕРЕСЫ ЧИТАТЕЛЯ по разделам. Что означают уровни:
  ОЧЕНЬ ИНТЕРЕСНО — присылай регулярно, включая события средней значимости;
  ИНТЕРЕСНО — присылай, когда есть заметное событие;
  СРЕДНИЙ ИНТЕРЕС — только значимые новости;
  СЛАБЫЙ ИНТЕРЕС — только очень крупные события.
Темы, которых нет в списке, — на твоё усмотрение с учётом профиля
читателя, но только заметные события.

{topics_text(config)}

НИКОГДА не присылай:
{never}

ПРАВИЛА ОТБОРА:
{rules_text}

{breaking}
{strict}
КАК ЧИТАТЕЛЬ ЧИТАЕТ НОВОСТИ — в четыре шага, и каждый следующий шаг
даёт НОВУЮ информацию, не повторяя предыдущий:
  1) заголовок — понять, о чём событие;
  2) текст под заголовком — если зацепило: где, когда, кто, почему;
  3) раскрывающийся блок — если интересно: полный конспект статьи;
  4) ссылка на оригинал — только чтобы углубиться. К этому моменту
     читатель должен уже знать почти всё ключевое из статьи.

ЗАДАЧА:
1. send — подходит ли новость читателю.
2. topic — ровно одно название РАЗДЕЛА из списка выше, дословно,
   например «Технологии и ИИ». Если ни один не подходит — «Прочее».
3. country — страна ИЛИ регион, ГДЕ ПРОИСХОДИТ СОБЫТИЕ, по-русски.
   Это НЕ страна издания. Примеры: «Израиль», «США», «Казахстан»,
   «Газа», «Китай». Если событие охватывает несколько стран — укажи
   главную или регион: «Ближний Восток», «ЕС», «Мир».
4. importance — ровно одно слово: СРОЧНО или ОБЫЧНО.
   СРОЧНО ставится КРАЙНЕ РЕДКО — не чаще одной новости из тридцати.
   Это событие, о котором человек захочет узнать немедленно и которое
   меняет положение дел:
     - начало войны, вторжение, массированный удар по стране;
     - заключение или срыв перемирия в крупном конфликте;
     - смерть, убийство, свержение или отставка главы государства;
     - крупный теракт с множеством жертв;
     - обвал или скачок рынков, дефолт страны, резкая девальвация;
     - решение, прямо и серьёзно затрагивающее Казахстан.
   ВО ВСЕХ ОСТАЛЬНЫХ СЛУЧАЯХ — ОБЫЧНО.

5. title_ru — ШАГ 1, ЗАГОЛОВОК. Строго не длиннее {title_max} знаков
   вместе с пробелами (примерно 5–7 слов) — так он уместится в две
   строки на телефоне. Только суть: что случилось или кто что сделал.
   Без подробностей, без точки в конце.
   Это не перевод, а самостоятельный заголовок, как у редактора
   русскоязычного издания: живой, естественный, читается с первого раза.
   - Не переводи английские обороты буквально. «Officials said» — это
     не «официальные лица сказали», а «власти сообщили».
   - Не нагромождай существительные в родительном падеже. Плохо:
     «Усиление влияния ультраправых сил Европы». Хорошо: «Ультраправые
     в Европе набирают силу».
   - Глаголы вместо отглагольных существительных.
   - Имена — в принятой русской форме: Нетаньяху, Эрдоган.
   - Без английских слов, кроме названий компаний и изданий.

6. lead — ШАГ 2, ТЕКСТ ПОД ЗАГОЛОВКОМ. Строго не длиннее {lead_max}
   знаков вместе с пробелами — это 1–2 коротких предложения.
   Отвечает на вопросы, на которые НЕ отвечает заголовок: где именно,
   когда, кто участники, из-за чего или почему, главная цифра.
   НЕ ПОВТОРЯЙ заголовок ни словами, ни смыслом — читатель его уже
   прочитал. Каждое слово должно добавлять новое.
   Пример. Заголовок: «Израиль ударил по пригороду Бейрута».
     Плохо: «Израиль нанёс удар по пригороду Бейрута. Удар был
     направлен против Хезболлы.» — повтор заголовка.
     Хорошо: «19 сентября, район Дахия. Целью был командир Хезболлы —
     Израиль называет это ответом на обстрел севера страны.»

7. detail — ШАГ 3, РАСКРЫВАЮЩИЙСЯ БЛОК: полный конспект статьи,
   {detail_len}. После него читатель должен знать почти всё ключевое,
   что есть в статье. Абзацы разделяй пустой строкой.
   Включи всё конкретное, что есть в тексте:
     - главные факты в порядке важности;
     - имена людей и их должности;
     - названия организаций, компаний, ведомств, мест;
     - все значимые числа: суммы, проценты, количество, сроки, даты;
     - суть заявлений: кто что именно потребовал, пообещал, отверг;
       позиции разных сторон, если они есть;
     - коротко контекст (1–2 предложения): почему это происходит,
       что было до этого;
     - что будет дальше и когда.
   НЕ повторяй lead дословно — развивай его. Если в тексте статьи
   фактов мало — пиши короче, но только по существу.

ПРАВИЛА НАПИСАНИЯ:
{writing_text}
- Живой русский язык: короткие фразы, глаголы вместо канцелярита,
  никакой кальки с английского.
- Не переводи дословно и не копируй фразы из оригинала.
- Без вводных вроде «В статье говорится» и без названия издания.

ГЛАВНОЕ ПРАВИЛО ТОЧНОСТИ:
Пиши ТОЛЬКО то, что прямо сказано в заголовке и тексте статьи выше.
- Если смысл фразы не до конца ясен — передай её осторожно и общими
  словами, но НИКОГДА не угадывай. Выдуманная деталь хуже, чем её
  отсутствие.
- Не добавляй фактов, которых нет в исходнике: ни причин, ни цифр,
  ни имён, ни оценок, ни последствий. Исключение — пояснение
  экономического термина по правилам написания.
- Не пиши фраз-заполнителей вроде «эксперты отмечают», «это вызывает
  обеспокоенность», «это подчёркивает растущую роль». Лучше меньше,
  но по существу.

Если новость не подходит — send: false, остальные поля пустые строки."""


def ask_gemini(item, config):
    """Спрашивает Gemini. Если модель перегружена — пробует следующую.
    Возвращает словарь, {} при сбое, None если исчерпан дневной лимит."""
    payload = {
        "contents": [{"parts": [{"text": build_prompt(item, config)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 3000,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    for model in GEMINI_MODELS:
        for attempt in range(2):
            try:
                resp = requests.post(
                    gemini_url(model), headers=headers, json=payload, timeout=60
                )
            except Exception as e:
                print(f"[GEMINI/{model}] сеть: {type(e).__name__}")
                break  # к следующей модели

            if resp.status_code == 503:
                if attempt == 0:
                    time.sleep(5)
                    continue
                print(f"[GEMINI/{model}] перегружена, пробую следующую модель")
                break

            if resp.status_code == 429:
                print(f"[GEMINI/{model}] лимит модели исчерпан, пробую следующую")
                break

            if not resp.ok:
                print(f"[GEMINI/{model}] ошибка {resp.status_code}: {resp.text[:150]}")
                break

            try:
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw)
            except Exception as e:
                print(f"[GEMINI/{model}] не разобрал ответ: {e}")
                return {}

    print(f"[GEMINI] все модели недоступны, пропускаю «{item['title'][:45]}»")
    return {}


# ---------- Оформление ----------

def fit_lead(lead, detail, max_chars):
    """Подгоняет текст под заголовком под заданную длину. Лишние
    предложения переносит в начало раскрывающейся части, чтобы текст
    не обрывался на полуслове."""
    lead = (lead or "").strip()
    detail = (detail or "").strip()
    if len(lead) <= max_chars:
        return lead, detail

    # режем по границам предложений
    sentences = re.split(r"(?<=[.!?…])\s+", lead)
    kept, moved = [], []
    used = 0
    for s in sentences:
        if not moved and (not kept or used + len(s) + 1 <= max_chars):
            kept.append(s)
            used += len(s) + 1
        else:
            moved.append(s)

    # Если даже одно предложение длиннее лимита — оставляем его целиком.
    # Лучше лишняя строка, чем оборванная на полуслове фраза.
    new_lead = " ".join(kept).strip()
    head = " ".join(moved).strip()
    new_detail = f"{head}\n\n{detail}".strip() if head else detail
    return new_lead, new_detail


def trim_text(text, limit):
    """Обрезает слишком длинный текст по границе предложения.
    Страховка от лимита Telegram (4096 знаков на сообщение)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    pos = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: pos + 1] if pos > limit // 2 else cut.rstrip() + "…"


def esc(text):
    return html.escape(str(text or ""), quote=False)


def make_hashtag(text):
    """Превращает «Санкции и торговые войны» в #санкции_и_торговые_войны."""
    text = re.sub(r"\([^)]*\)", "", text)          # убрать скобки
    text = re.sub(r"[^\w\s]", "", text, flags=re.U)  # убрать знаки
    words = text.lower().split()
    if not words:
        return ""
    return "#" + "_".join(words[:4])


def pick_emoji(topic, config):
    """Эмодзи раздела. Если модель назвала раздел неточно — ищем
    раздел, чьё название входит в ответ или наоборот."""
    sections = config.get("topics") or {}
    default = config.get("default_emoji", "📰")
    topic = (topic or "").strip()
    if topic in sections:
        return (sections[topic] or {}).get("emoji", default)
    low = topic.lower()
    for name, body in sections.items():
        n = name.lower()
        if low and (n in low or low in n):
            return (body or {}).get("emoji", default)
    return default


def format_message(item, data, config):
    """Собирает сообщение из четырёх уровней:
      1) заголовок  2) текст под ним  3) раскрывающийся конспект
      4) подпись в одну строку: источник-ссылка +N · страна"""
    lead_max = config.get("lead_max_chars", 125)
    topic = data.get("topic", "")
    emoji = pick_emoji(topic, config)
    title = esc(data.get("title_ru", "")).strip()
    raw_lead, raw_detail = fit_lead(
        data.get("lead", ""), data.get("detail", ""), lead_max
    )
    lead = esc(raw_lead)
    detail = esc(trim_text(raw_detail, 3000))
    country = (data.get("country") or "").strip()
    importance = (data.get("importance") or "").strip().upper()

    parts = []

    # Пометка ставится ТОЛЬКО для по-настоящему срочных новостей.
    # Отдельной строкой с пустой строкой после неё — иначе два эмодзи
    # подряд ломают отображение текста в Telegram.
    if importance == "СРОЧНО":
        parts.append("🔴 <b>СРОЧНО</b>")
        parts.append("")

    parts.append(f"{emoji} <b>{title}</b>" if title else f"{emoji} <b>{esc(topic)}</b>")
    parts.append("")
    parts.append(lead)

    # Раскрывающийся блок с подробностями
    if detail:
        parts.append("")
        parts.append(f"<blockquote expandable>{detail}</blockquote>")

    parts.append("")

    if config.get("show_hashtags", False):
        tags = [t for t in (make_hashtag(topic), make_hashtag(country)) if t]
        if tags:
            parts.append(esc(" ".join(tags)))

    # Подпись: название источника само является ссылкой на статью,
    # +N — сколько ещё изданий написали об этом же событии.
    link = html.escape(item["link"], quote=True)
    signature = f'<a href="{link}">{esc(item["source"])}</a>'
    extra = item.get("sources_count", 1) - 1
    if extra > 0:
        signature += f" +{extra}"
    if country:
        signature += f"  ·  {esc(country)}"
    parts.append(signature)

    return "\n".join(parts)


# ---------- Погода ----------

WEATHER_TEXT = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность",
    3: "пасмурно", 45: "туман", 48: "туман с изморозью",
    51: "лёгкая морось", 53: "морось", 55: "сильная морось",
    56: "ледяная морось", 57: "ледяная морось",
    61: "небольшой дождь", 63: "дождь", 65: "сильный дождь",
    66: "ледяной дождь", 67: "ледяной дождь",
    71: "небольшой снег", 73: "снег", 75: "сильный снег", 77: "снежная крупа",
    80: "ливни", 81: "ливни", 82: "сильные ливни",
    85: "снегопад", 86: "сильный снегопад",
    95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом",
}

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]


def weather_emoji(code):
    if code in (0, 1):
        return "☀️"
    if code == 2:
        return "⛅"
    if code == 3:
        return "☁️"
    if code in (45, 48):
        return "🌫"
    if code >= 95:
        return "⛈"
    if 71 <= code <= 77 or code in (85, 86):
        return "🌨"
    return "🌧"


def fmt_temp(t):
    t = round(t)
    return "0°" if t == 0 else f"{t:+d}°"


def build_weather_message(hourly, local_date, utc_offset, city):
    """Собирает пост о погоде из почасового прогноза Open-Meteo
    (время в прогнозе — по Гринвичу, переводим в местное)."""
    rows = []
    for i, ts in enumerate(hourly.get("time", [])):
        t_local = datetime.fromisoformat(ts) + timedelta(hours=utc_offset)
        if t_local.date() == local_date:
            rows.append((t_local.hour, i))
    if not rows:
        return ""

    def at(hour, key):
        for h, i in rows:
            if h == hour:
                return hourly[key][i]
        return None

    temps = [(label, at(h, "temperature_2m"))
             for label, h in (("утром", 8), ("днём", 14), ("вечером", 20))]
    temps = [(l, t) for l, t in temps if t is not None]
    day = [i for h, i in rows if 8 <= h <= 21]
    codes = [hourly["weather_code"][i] for i in day
             if hourly["weather_code"][i] is not None]
    rain = [hourly["precipitation_probability"][i] for i in day
            if hourly["precipitation_probability"][i] is not None]
    wind = [hourly["wind_speed_10m"][i] for i in day
            if hourly["wind_speed_10m"][i] is not None]

    code = max(codes) if codes else 0
    sky = WEATHER_TEXT.get(code, "")
    rain_max = max(rain) if rain else 0
    wind_max = round(max(wind)) if wind else 0

    title = (f"{weather_emoji(code)} <b>Погода в {esc(city)} · "
             f"{local_date.day} {MONTHS_GEN[local_date.month - 1]}</b>")
    line1 = ", ".join(f"{l} {fmt_temp(t)}" for l, t in temps)
    line1 = line1[:1].upper() + line1[1:] + "."
    rain_txt = ("без осадков" if rain_max < 20
                else f"вероятность осадков {rain_max}%")
    line2 = f"{sky[:1].upper() + sky[1:]}, {rain_txt}. Ветер до {wind_max} км/ч."
    return "\n".join([title, "", line1, line2])


def post_weather(config, state):
    """Раз в день утром отправляет короткий прогноз погоды."""
    w = config.get("weather") or {}
    if not w.get("enabled"):
        return
    offset = config.get("utc_offset", w.get("utc_offset", 5))
    now_local = datetime.now(timezone.utc) + timedelta(hours=offset)
    today = now_local.strftime("%Y-%m-%d")
    if state.get("weather_date") == today:
        return
    if not (w.get("post_after_hour", 7) <= now_local.hour
            < w.get("post_before_hour", 12)):
        return
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": w.get("latitude"),
                "longitude": w.get("longitude"),
                "hourly": "temperature_2m,precipitation_probability,"
                          "weather_code,wind_speed_10m",
                "timezone": "GMT",
                "forecast_days": 2,
            },
            timeout=20,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
    except Exception as e:
        print(f"[ПОГОДА] не удалось получить прогноз: {type(e).__name__}")
        return
    text = build_weather_message(hourly, now_local.date(), offset,
                                 w.get("city", ""))
    if text and send_telegram(text):
        state["weather_date"] = today
        print("[ПОГОДА] прогноз отправлен")


# ---------- Основной цикл ----------

def local_now(config):
    offset = config.get("utc_offset", (config.get("weather") or {}).get("utc_offset", 5))
    return datetime.now(timezone.utc) + timedelta(hours=offset)


def in_quiet_hours(config):
    """Ночью новости не присылаются. Расписание и так не запускает бота
    в это время, но GitHub иногда запускает с опозданием — эта проверка
    не даёт опоздавшему запуску прислать новости после полуночи."""
    q = config.get("quiet_hours") or {}
    if not q:
        return False
    start, end = q.get("from", 0), q.get("to", 5)
    h = local_now(config).hour
    return start <= h < end if start < end else (h >= start or h < end)


def build_queue(unique, kz_share):
    """Очередь на обработку: мировые и казахстанские новости вперемешку
    в заданной пропорции. Внутри каждой группы — по убыванию доверия.
    При доле 0.3 получается порядок: мир, мир, КЗ, мир, мир, КЗ...
    Без этого казахстанские издания (доверие 5-7) всегда оказывались
    в конце очереди и не успевали попасть в отправку."""
    world = [i for i in unique if not i.get("kz")]
    kz = [i for i in unique if i.get("kz")]
    queue, k = [], 0
    while world or kz:
        n = len(queue) + 1
        want_kz = kz_share > 0 and (k + 1) / n <= kz_share + 0.05
        if kz and (want_kz or not world):
            queue.append(kz.pop(0))
            k += 1
        else:
            queue.append(world.pop(0))
    return queue


def main():
    config = load_config()
    if in_quiet_hours(config):
        print("Тихие часы — новости не отправляю.")
        return

    if not check_telegram():
        print("Останавливаюсь: сначала почините доступ в Telegram (см. выше).")
        return

    memory_hours = config.get("duplicate_memory_hours", 48)
    state = load_state()

    # Погода — раз в день утром, отдельным постом
    post_weather(config, state)

    seen = set(state["seen"])
    new_ids = []
    sent = 0
    kz_sent = 0

    send_limit = config.get("max_per_run", 0) or 10**9
    ai_limit = config.get("max_ai_calls_per_run", 30)
    max_age = config.get("max_age_hours", 4)
    pause = config.get("seconds_between_ai_calls", 5)
    threshold = config.get("duplicate_threshold", 0.5)
    fetch_full = config.get("fetch_full_text", True)
    min_desc = config.get("min_description_chars", 1500)
    attempts = config.get("max_text_attempts", 3)
    kz_share = config.get("kz_share", 0)
    kz_limit = max(1, round(send_limit * kz_share)) if kz_share else 10**9
    patterns = build_keyword_patterns(config.get("keywords", []))
    blocked_list = config.get("blocked_sources", [])
    ru_dups = 0
    enriched = 0
    swapped = 0
    kz_skipped = 0

    # Сбор
    # ленты качаем параллельно — так сбор занимает секунды, а не минуту
    items = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for got, log in pool.map(fetch_rss, config.get("world_news_rss", [])):
            print(log)
            items += got

    unseen = []
    blocked_n = 0
    for i in items:
        if i["id"] in seen:
            continue
        if is_blocked(i, blocked_list):
            blocked_n += 1
            new_ids.append(i["id"])
            continue
        unseen.append(i)
    print(f"Всего получено: {len(items)}, новых: {len(unseen)}"
          f" (отсеяно нежелательных изданий: {blocked_n})")

    # ШАГ 1 — свежесть
    fresh = []
    for i in unseen:
        if is_fresh(i, max_age):
            fresh.append(i)
        else:
            new_ids.append(i["id"])
    print(f"Свежих (моложе {max_age} ч): {len(fresh)}")

    # ШАГ 2 — ключевые слова
    candidates = []
    for i in fresh:
        if passes_keywords(i, patterns):
            candidates.append(i)
        else:
            new_ids.append(i["id"])
    print(f"Прошло отсев по словам: {len(candidates)}")

    # ШАГ 3 — дедупликация
    unique, duplicates = deduplicate(candidates, threshold, state["recent"])
    for i in duplicates:
        new_ids.append(i["id"])
    print(f"После склейки дублей осталось: {len(unique)} (убрано {len(duplicates)})")

    # порядок: самые авторитетные источники первыми, но мировые
    # и казахстанские новости вперемешку — в пропорции kz_share
    unique.sort(key=lambda i: -i["trust"])
    batch = build_queue(unique, kz_share)[:ai_limit]
    print(f"Пойдёт в Gemini: до {len(batch)}"
          f" (из них казахстанских: {sum(1 for i in batch if i.get('kz'))})")

    # ШАГ 4 — полный текст статей. Качаем параллельно, чтобы не
    # растягивать запуск: каждая статья — это 1-3 обращения к сайтам.
    chosen = {}
    if fetch_full and batch:
        def work(it):
            return pick_best_text(it["group"], min_desc, blocked_list, attempts)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, batch))
        for it, (best, fetched) in zip(batch, results):
            chosen[id(it)] = best
            enriched += fetched

    # ШАГ 5-6 — ИИ и отправка
    ai_used = 0
    last_call = 0.0
    for item in batch:
        if sent >= send_limit:
            break

        group = item["group"]
        best = chosen.get(id(item), item) if fetch_full else item
        if best is None:
            # у всех изданий сюжета настоящий адрес оказался заблокированным
            new_ids.append(item["id"])
            continue
        if best is not item:
            swapped += 1

        # баланс Казахстан / мир: лимит на казахстанские новости за запуск.
        # Не помечаем как просмотренное — может уйти в следующем запуске.
        if best.get("kz") and kz_sent >= kz_limit:
            kz_skipped += 1
            continue

        # выдерживаем промежуток между обращениями к ИИ
        wait = pause - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()

        data = ask_gemini(best, config)
        ai_used += 1

        if data is None:
            break

        new_ids.append(item["id"])

        if not (data.get("send") and data.get("lead")):
            continue

        is_kz = best.get("kz") or (data.get("country") or "").startswith("Казахстан")
        if is_kz and kz_sent >= kz_limit:
            kz_skipped += 1
            continue

        # второй этап склейки — уже по русскому тексту
        repeat, ru_tokens = is_repeat_ru(data, state["recent_ru"], threshold)
        if repeat:
            ru_dups += 1
            print(f"  дубль по смыслу, не шлю: «{data.get('title_ru','')[:50]}»")
            continue

        best["sources_count"] = len(group)
        if send_telegram(format_message(best, data, config)):
            sent += 1
            if is_kz:
                kz_sent += 1
            state["recent"].append(
                {"tokens": sorted(item["tokens"]), "ts": time.time()}
            )
            state["recent_ru"].append(
                {"tokens": sorted(ru_tokens), "ts": time.time()}
            )

    state["seen"] = state["seen"] + new_ids
    save_state(state, memory_hours)
    print(
        f"Обращений к Gemini: {ai_used}. "
        f"Догружено статей: {enriched}. "
        f"Текст взят у другого издания сюжета: {swapped}. "
        f"Отсеяно как дубль после перевода: {ru_dups}. "
        f"Отложено по балансу Казахстан/мир: {kz_skipped}. "
        f"Отправлено сообщений: {sent} (из них о Казахстане: {kz_sent})"
    )


if __name__ == "__main__":
    main()
