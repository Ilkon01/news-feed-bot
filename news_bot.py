"""
Персональный новостной бот (бесплатная версия на Google Gemini).

Как работает:
  ШАГ 1 — берём только свежие новости (по времени публикации)
  ШАГ 2 — отсев по ключевым словам (бесплатно, без ИИ)
  ШАГ 3 — дедупликация: одинаковые новости из разных источников
          склеиваются в один сюжет
  ШАГ 3а — Gemini одним запросом оценивает кандидатов по интересам
          читателя (0–10) и отмечает повторы уже отправленного;
          дальше идут только лучшие
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
        "people": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "send", "topic", "country", "importance", "title_ru", "lead", "detail",
        "people",
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
    raw = name.strip()
    if re.match(r"https?://", raw):                    # имя — это адрес
        raw = domain_of(raw)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", raw)       # хвост в скобках
    name = re.split(r"\s+[-–|:]\s+", name)[0]          # всё после тире
    short = re.sub(r"\s*\.(com|net|org|ru|kz|kg|uz)\s*$", "", name, flags=re.I)
    # «24.kz» не превращать в «24»: без зоны имя теряет смысл
    if len(short.strip()) >= 3 and not short.strip().isdigit():
        name = short
    return name.strip().rstrip("/")[:40]


def domain_of(url):
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1).lower().replace("www.", "") if m else ""


def is_blocked(item, blocked):
    """Отсев нежелательных изданий по домену или названию.
      .ru          — зона целиком;
      rt.com       — домен и его поддомены (но не sport.com);
      tass, медуза — часть адреса или названия издания."""
    dom = domain_of(item.get("link", ""))
    src_name = (item.get("source") or "").lower()
    for b in blocked:
        b = b.lower().strip()
        if not b:
            continue
        if b.startswith("."):          # зона целиком, например .ru
            if dom.endswith(b):
                return True
        elif "." in b and " " not in b:  # конкретный домен
            if dom == b or dom.endswith("." + b) or src_name == b:
                return True
        elif b in dom or b in src_name:
            return True
    return False


def is_blocked_url(item, parts):
    """Разделы сайтов, которые не нужны: местные новости Австралии
    и британская внутренняя политика у Guardian, Гонконг у SCMP."""
    link = (item.get("link") or "").lower()
    return any(p.lower() in link for p in parts if p.strip())


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
    """Доля общих слов от более короткого заголовка."""
    if not tokens_a or not tokens_b:
        return 0.0
    common = len(tokens_a & tokens_b)
    return common / min(len(tokens_a), len(tokens_b))


def is_sent_before(tokens, sent_sets, threshold, min_common):
    """Совпадает ли сюжет с уже отправленным. Кроме доли общих слов
    нужно ещё и минимум min_common общих слов: иначе «Israel strikes
    Beirut» и «Israel strikes Gaza» (общие только israel и strike)
    считались одной новостью, и вторая пропадала на трое суток."""
    for rs in sent_sets:
        if (len(tokens & rs) >= min_common
                and similarity(tokens, rs) >= threshold):
            return True
    return False


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


def deduplicate(items, threshold, recent, min_common=3):
    """Оставляет по одной новости на сюжет — от источника с высшим доверием.
    Также убирает то, что уже отправлялось в прошлых запусках."""
    for item in items:
        item["tokens"] = title_tokens(item["title"])

    # 1) убрать совпадающее с уже отправленным ранее. Перефразированный
    # дубль, который здесь проскочит, поймает сверка по русскому тексту.
    recent_sets = [set(r["tokens"]) for r in recent]
    not_repeated, repeated = [], []
    for item in items:
        if is_sent_before(item["tokens"], recent_sets, threshold, min_common):
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

    today = local_now(config)
    today_str = f"{today.day} {MONTHS_GEN[today.month - 1]} {today.year}"

    return f"""Ты — персональный новостной редактор. Решаешь, нужна ли эта
новость читателю, и если да — пишешь её для Telegram-канала.

СЕГОДНЯ: {today_str}. Твои собственные знания о мире УСТАРЕЛИ.
Кто сейчас президент, премьер, министр или глава компании — бери
ТОЛЬКО из текста статьи ниже, НИКОГДА из своей памяти. Если в статье
сказано «Трамп», нельзя писать «Байден», и наоборот.

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
   Раздел «Ислам и мусульманский мир» — ТОЛЬКО для новостей, где речь
   прямо об исламе, мусульманах или движении Хизмет. Новости о
   христианстве, Ватикане, иудаизме и других религиях — раздел
   «Другие религии». Обычная политика в мусульманской стране без
   религиозной стороны — «Мировая политика».
3. country — страна события, по-русски. Это НЕ страна издания.
   ОБЫЧНО — ОДНА страна, где происходит событие: «США», «Казахстан»,
   «Китай», «Газа». Если событие охватывает много стран — регион:
   «Ближний Восток», «ЕС», «Мир».
   ДВЕ СТРАНЫ через косую черту — ТОЛЬКО когда событие прямо
   связывает две конкретные страны: одна действует в отношении
   другой, между ними конфликт, переговоры, сделка, санкции.
   Формат: «ГДЕ ПРОИСХОДИТ/О КАКОЙ СТРАНЕ РЕЧЬ».
     «США/Палестина» — Вашингтон принял решение по Палестине;
     «Россия/Украина» — удар России по Украине;
     «Казахстан/Китай» — в Астане подписали соглашение с Китаем;
     «Турция/Германия» — Эрдоган принял немецкого премьера в Анкаре.
   НЕ ставь вторую страну, если она лишь упомянута мимоходом или
   событие внутреннее. «Выборы в Германии» — просто «Германия»;
   «Пожар в Анкаре» — просто «Турция». Никогда не больше двух стран.
   Спорт — только страна, где проходят соревнования, без гражданства
   спортсменов: «Испанец проиграл бельгийцу в Шанхае» — «Китай».
   Санкции и решения — кто принял/в отношении кого: «США сняли
   санкции с российского дизеля» — «США/Россия», а не «Россия/Украина».
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
   ВО ВСЕХ ОСТАЛЬНЫХ СЛУЧАЯХ — ОБЫЧНО. Смена президента небольшой
   страны, обычное падение акций, рядовой удар в идущей войне —
   ОБЫЧНО.

5. title_ru — ШАГ 1, ЗАГОЛОВОК. 4–7 слов, примерно до {title_max}
   знаков — так он уместится в две строки на телефоне.
   - ВСЕГДА только целыми словами. Никогда не обрывай слово: если
     заголовок длинный — перефразируй короче, а не обрезай.
   - Заголовок должен быть понятен САМ ПО СЕБЕ, без текста ниже:
     КТО сделал ЧТО. Если прочитав его, нельзя понять, что произошло, —
     это плохой заголовок.
     Плохо: «СМИ и Трамп: сети ждут новостей» — что случилось, непонятно.
     Плохо: «Anthropic предупреждает об экзистенциальных ри» — оборван.
     Хорошо: «Anthropic назвала риски ИИ в документах к IPO».
   Без подробностей, без точки в конце.
   Это не перевод, а самостоятельный заголовок, как у редактора
   русскоязычного издания: живой, естественный, читается с первого раза.
   - Не переводи английские обороты буквально. «Officials said» — это
     не «официальные лица сказали», а «власти сообщили».
   - Не нагромождай существительные в родительном падеже. Плохо:
     «Усиление влияния ультраправых сил Европы». Хорошо: «Ультраправые
     в Европе набирают силу».
   - Глаголы вместо отглагольных существительных.
   - Имена — в принятой русской форме: Нетаньяху, Эрдоган. Никогда
     не латиницей: «Дональд Трамп», а не «Дональд Trump».
   - Без английских слов, кроме названий компаний и изданий.
   - Если глаголу нужно дополнение — оно должно быть в заголовке.
     Плохо: «Токаев и монархи Бельгии посетили» — посетили что?

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
   Каждое предложение detail — НОВЫЙ факт, которого нет ни в
   заголовке, ни в lead. Пересказ уже сказанного другими словами —
   это повтор, он запрещён.
   Пример повтора, так делать НЕЛЬЗЯ:
     Заголовок: «Palo Alto Networks запускает ИИ-сервис»
     lead: «Новая платформа кибербезопасности создана на базе моделей
     Claude и GPT.»
     detail: «Компания Palo Alto Networks представила новый сервис
     кибербезопасности на основе ИИ. В основе используются модели
     Claude и GPT.» — всё это уже сказано выше.
   Если в тексте статьи нет новых фактов для detail — оставь detail
   пустым. Пустой detail лучше повтора.

8. people — список ВСЕХ людей, которых ты упомянул в title_ru, lead
   и detail, в том написании, как они указаны в тексте статьи
   (латиницей, если статья на английском): ["Donald Trump", "Xi
   Jinping"]. Если людей нет — пустой список. Бот сверит каждое имя
   с текстом статьи.

САМОПРОВЕРКА ПЕРЕД ОТВЕТОМ:
- В заголовке все слова целые, и из него ясно, что произошло?
- Каждое имя в тексте действительно есть в статье?
- В lead нет того, что уже сказано в заголовке?
- В detail нет того, что уже сказано в заголовке или lead?

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
  обеспокоенность», «это подчёркивает растущую роль», «продолжает
  привлекать внимание». Если в статье мнение высказал конкретный
  человек или организация — назови их: «Аналитик Capital Economics
  Джон Смит считает…». Безымянных «экспертов» не бывает.
- Весь текст — по-русски. Английские фразы из статьи переводи.
- НИКОГДА не пиши о том, чего в статье НЕТ: «подробности не
  приводятся», «другие детали не сообщаются», «компания не раскрыла
  сроки». Читателю это ничего не даёт.
- Не упоминай, откуда взята новость: «об этом сообщило агентство»,
  «как стало известно из публикации».

ЕСЛИ МАТЕРИАЛА МАЛО. Если в тексте статьи нет конкретики — ни имён,
ни цифр, ни деталей, только общая фраза, — ставь send: false.
Исключение — события уровня СРОЧНО.

Если новость не подходит — send: false, остальные поля пустые строки."""


def response_schema(config):
    """Схема ответа. Раздел — строго из списка: раньше модель писала
    «Экономика Казахстана» вместо «Казахстан: экономика и кошелёк»,
    и у 5% новостей вместо значка раздела стоял 📰."""
    schema = json.loads(json.dumps(RESPONSE_SCHEMA))
    sections = list((config.get("topics") or {}).keys())
    if sections:
        schema["properties"]["topic"]["enum"] = sections + ["Прочее"]
    return schema


def ask_gemini(item, config, fix_note=""):
    """Пишет новость. fix_note — замечания к прошлому ответу, если его
    пришлось вернуть на доработку. Возвращает словарь, {} при сбое."""
    prompt = build_prompt(item, config)
    if fix_note:
        prompt += (
            "\n\nТВОЙ ПРОШЛЫЙ ОТВЕТ НА ЭТУ НОВОСТЬ ОТКЛОНЁН. Ошибки:\n"
            + fix_note
            + "\nНапиши ответ заново, исправив эти ошибки."
        )
    data, model = call_gemini(prompt, response_schema(config))
    if not isinstance(data, dict):
        print(f"[GEMINI] все модели недоступны, пропускаю «{item['title'][:45]}»")
        return {}
    data["_model"] = model
    return data


def call_gemini(prompt, schema, temperature=0.2):
    """Обращение к Gemini. Если модель перегружена или ответ не
    разобрался — пробует следующую. Возвращает (ответ, модель)
    или (None, None) при сбое всех моделей."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            # С запасом: у «думающих» моделей размышления входят в этот
            # лимит, и при 3000 ответ мог обрываться посреди JSON.
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": schema,
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
                cand = resp.json()["candidates"][0]
                raw = cand["content"]["parts"][0]["text"]
                data = json.loads(raw)
                expected = list if schema.get("type") == "ARRAY" else dict
                if not isinstance(data, expected):
                    raise ValueError("ответ не того вида")
                return data, model
            except Exception as e:
                reason = ""
                try:
                    reason = resp.json()["candidates"][0].get("finishReason", "")
                except Exception:
                    pass
                print(f"[GEMINI/{model}] не разобрал ответ"
                      f"{' (' + reason + ')' if reason else ''}: "
                      f"{type(e).__name__}, пробую следующую модель")
                break

    return None, None


# ---------- Оценка кандидатов по интересам ----------

RANK_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "i": {"type": "INTEGER"},
            "score": {"type": "INTEGER"},
            "repeat": {"type": "BOOLEAN"},
        },
        "required": ["i", "score", "repeat"],
    },
}


def build_rank_prompt(pool, recent_titles, config):
    """Один запрос на всех кандидатов: насколько каждая новость нужна
    читателю и не повторяет ли она уже отправленное."""
    lines = []
    for n, it in enumerate(pool):
        summary = re.sub(r"\s+", " ", it.get("text") or "")[:220]
        lines.append(f"{n}. [{it['source']}] {it['title']} — {summary}")
    recent = "\n".join(f"- {t}" for t in recent_titles) or "- (пока ничего)"
    today = local_now(config)
    return f"""Ты — редактор личного новостного канала. Оцени каждую новость
из списка: насколько она нужна ЭТОМУ читателю. Сегодня {today.day}
{MONTHS_GEN[today.month - 1]} {today.year}.

О ЧИТАТЕЛЕ:
{bullets(config.get("reader_profile"))}

ИНТЕРЕСЫ ЧИТАТЕЛЯ по разделам и уровням:
{topics_text(config)}

НИКОГДА не присылать:
{bullets(config.get("never_send"))}

ПРАВИЛА ОТБОРА:
{bullets(config.get("editorial_rules"))}

ШКАЛА score от 0 до 10. Будь строгим: высокий балл — исключение.
Обычно 8 и выше заслуживают не больше 10–15 новостей из списка,
а большинство получает 3–6.

СНАЧАЛА проверь, не относится ли новость к списку ниже — тогда 0–3,
какой бы важной ни была тема:
  - рядовые сводки боёв и ударов без изменения положения дел, даже
    с жертвами: «Авиаудар убил троих детей в Йемене», «Палестинец
    погиб при ударе у школы в Газе» — 2. Высокий балл — только
    начало или конец войны, перемирие, переговоры, удар, меняющий
    ход событий (например, закрытие Ормузского пролива);
  - местный криминал и происшествия в любой стране, включая
    Казахстан: кражи, драки, ДТП, бытовые убийства, аресты за мелкие
    правонарушения, отдельные несчастные случаи;
  - внутренняя политика и местные новости Австралии, Великобритании,
    Гонконга и других стран без значения для мира или Казахстана;
  - протокольные визиты, встречи и меморандумы без конкретных
    решений и цифр;
  - анонсы матчей, слухи и обзоры игр, реклама, светская хроника.

Остальное оцени так:
  9–10 — тема «очень интересно» и заметное событие; любые новости
         о движении Хизмет и Фетхуллахе Гюлене; решения о хадже
         и умре — правила, квоты, цены, визы, политика Саудовской
         Аравии; крупные события в Казахстане;
  7–8  — тема «интересно» или содержательный материал из
         «очень интересно»; положение мусульман в мире — законы,
         запреты, дискриминация;
  4–5  — истории отдельных паломников, местные новости мечетей
         и общин без последствий для многих людей;
  5–6  — можно прислать, если нет лучшего;
  0–4  — не присылать.
Темы с уровнем «слабый интерес» — не выше 4, кроме очень крупных событий.

repeat: true — новость о том же событии, что уже отправлено
(список ниже), и не сообщает ничего существенно нового.

УЖЕ ОТПРАВЛЕНО:
{recent}

НОВОСТИ (номер. [издание] заголовок — начало текста):
{chr(10).join(lines)}

Верни по одному объекту на КАЖДЫЙ номер: i — номер, score, repeat."""


def rank_candidates(pool, recent_titles, config):
    """Возвращает {номер: (score, repeat)} или None при сбое Gemini."""
    data, _ = call_gemini(build_rank_prompt(pool, recent_titles, config),
                          RANK_SCHEMA, temperature=0.0)
    if not isinstance(data, list):
        return None
    out = {}
    for row in data:
        try:
            out[int(row["i"])] = (int(row["score"]), bool(row.get("repeat")))
        except Exception:
            continue
    return out


# ---------- Проверка ответа модели ----------

# Известные фигуры: как имя выглядит в русском тексте -> как оно может
# выглядеть в исходнике. Если модель написала имя, которого в статье
# нет, — это выдумка из её устаревшей памяти (например, «Байден»
# вместо «Трамп»).
# Имя ищется С ЗАГЛАВНОЙ буквы, а после основы — исключения для обычных
# слов: иначе «маски» принимались за Маска, «трамплин» — за Трампа,
# «мерцание» — за Мерца, и нормальная новость отклонялась.
KNOWN_PEOPLE = [
    ("Байден", r"Байден", ["biden", "байден"]),
    ("Трамп", r"Трамп(?!лин)", ["trump", "трамп"]),
    ("Харрис", r"Харрис", ["harris", "харрис"]),
    ("Вэнс", r"Вэнс", ["vance", "вэнс"]),
    ("Обама", r"Обам", ["obama", "обам"]),
    ("Путин", r"Путин", ["putin", "путин"]),
    ("Зеленский", r"Зеленск", ["zelensk", "зеленск"]),
    ("Си Цзиньпин", r"Цзиньпин", ["jinping", "цзиньпин", "xi"]),
    ("Нетаньяху", r"Нетаньяху", ["netanyahu", "нетаньяху"]),
    ("Эрдоган", r"Эрдоган", ["erdogan", "erdoğan", "эрдоган"]),
    ("Макрон", r"Макрон", ["macron", "макрон"]),
    ("Мерц", r"Мерц(?!ани|ающ|ает|ают|ал|ит|ат)", ["merz", "мерц"]),
    ("Шольц", r"Шольц", ["scholz", "шольц"]),
    ("Стармер", r"Стармер", ["starmer", "стармер"]),
    ("Сунак", r"Сунак", ["sunak", "сунак"]),
    ("Моди", r"Моди(?!ф)", ["modi", "моди"]),
    ("Токаев", r"Токаев", ["tokayev", "tokaev", "токаев"]),
    ("Назарбаев", r"Назарбаев", ["nazarbayev", "nazarbaev", "назарбаев"]),
    ("Мирзиёев", r"Мирзи[её]ев", ["mirziyoyev", "мирзиёев", "мирзиеев"]),
    ("Лукашенко", r"Лукашенк", ["lukashenk", "лукашенк"]),
    ("Гюлен", r"Гюлен", ["gulen", "gülen", "гюлен"]),
    ("Гутерриш", r"Гутерриш", ["guterres", "гутерриш"]),
    ("Маск", r"Маск(?!ир|от|и\b|ой\b|ами\b|ах\b)", ["musk", "маск"]),
    ("Альтман", r"Альтман", ["altman", "альтман"]),
    ("Амодеи", r"Амодеи", ["amodei", "амодеи"]),
]

# Слова, которыми заголовок может законно заканчиваться, хоть они
# и короткие: ИИ, ЕС, ООН и т.п. пишутся заглавными и проходят сами.
SHORT_OK = {"ии", "ес", "сша", "оон", "кнр", "рф", "рк", "ai", "ит"}

# Глаголы, после которых в заголовке обязано быть дополнение:
# «Токаев и монархи Бельгии посетили» — посетили что?
DANGLING_VERB = re.compile(
    r"(посетил|обсудил|подписал|представил|назвал|принял|провел|провёл"
    r"|заключил|объявил|утвердил|одобрил|запустил|раскрыл|получил"
    r"|выпустил|открыл|создал|предложил|ввел|ввёл|повысил|снизил)"
    r"(а|и|о)?")

# Известные люди латиницей посреди русского текста
LATIN_NAMES = re.compile(
    r"\b(Trump|Biden|Harris|Vance|Obama|Putin|Zelensk\w*|Netanyahu"
    r"|Erdogan|Erdoğan|Macron|Merz|Starmer|Modi|Tokayev|Musk|Altman"
    r"|Amodei|Guterres|Xi Jinping)\b")

# Отчество перед фамилией известного иностранца — выдумка модели
FOREIGN_PATRONYMIC = re.compile(
    r"\b[А-ЯЁ][а-яё]+(?:овн|евн|ович|евич)[а-яё]*\s+"
    r"(?:Трамп|Байден|Маск|Обам|Макрон|Мерц|Стармер|Нетаньяху|Эрдоган"
    r"|Кушнер|Харрис|Вэнс|Альтман|Цукерберг|Безос|Гейтс)[а-яё]*")

# Три и больше строчных английских слов подряд — непереведённый кусок.
# Названия (Rainbow Six Tactics) пишутся с заглавных и не ловятся.
ENGLISH_PHRASE = re.compile(r"\b[a-z]{3,}(?:\s+[a-z]{2,}){2,}\b")

# Признаки пустых фраз о том, чего в статье нет
FILLER_NEG = re.compile(
    r"\bне\s+(привод|раскры|уточн|сообщ|называ|указ|comment|disclos)", re.I)
FILLER_OBJ = re.compile(
    r"(подробност|детал|цифр|срок|стоимост|информаци|сумм)", re.I)
FILLER_SRC = re.compile(
    r"(стало известно из|(сообщ|подтверд|пиш|написа)\w* (агентств|издани"
    r"|газет|портал|телеканал)|опубликованн\w* "
    r"(агентств|издани)|по данным (агентства|издания)|в (статье|материале) "
    r"(говорится|не))", re.I)
# Безымянные «эксперты» и общие слова без фактов
FILLER_VAGUE = re.compile(
    r"(^|\s)((военные|правозащитники и|многие|некоторые)\s+)?"
    r"(эксперты|аналитики|специалисты|наблюдатели)\s+(отмечают|считают"
    r"|указывают|полагают|подчеркивают|подчёркивают)"
    r"|продолжает привлекать внимание", re.I)


def source_blob(item):
    return f"{item.get('title', '')} {item.get('text', '')}".lower()


TITLES = {"pope", "president", "prime", "minister", "king", "queen",
          "prince", "sheikh", "dr", "mr", "mrs", "папа", "президент",
          "король", "принц", "шейх", "премьер", "министр"}


def name_in_source(name, blob):
    """Есть ли человек в исходнике. Достаточно, чтобы нашлось любое из
    слов имени (кроме титулов) — по началу слова, чтобы «Trump's»
    и «Трампа» тоже находились. Проверка нарочно мягкая: её дело —
    поймать человека, которого в статье нет совсем."""
    words = [w for w in re.findall(r"[\w\-']+", name.lower())
             if len(w) >= 3 and w not in TITLES]
    if not words:
        return True
    # Имя по-русски, а статья на английском — сверить нельзя.
    # Известных людей в этом случае ловит список KNOWN_PEOPLE.
    if re.search(r"[а-яё]", name.lower()):
        cyr = len(re.findall(r"[а-яё]", blob))
        if cyr < len(blob) * 0.3:
            return True
    return any(re.search(r"\b" + re.escape(w[:5]), blob) for w in words)


def find_problems(data, item):
    """Ищет в ответе модели ошибки, из-за которых новость нельзя
    отправлять как есть. Возвращает список понятных описаний."""
    problems = []
    title = (data.get("title_ru") or "").strip()
    ru_text = f"{title} {data.get('lead', '')} {data.get('detail', '')}"
    blob = source_blob(item)

    # 1. Оборванный заголовок: последнее слово — обрубок
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]+", title)
    if words:
        last = words[-1]
        if (len(last) <= 2 and last.lower() == last
                and last.lower() not in SHORT_OK and not last.isdigit()):
            problems.append(
                f"заголовок оборван на «{last}» — нужен целый, "
                f"понятный заголовок из целых слов")
    if title.endswith(("-", "…", "...", ",", ":")):
        problems.append("заголовок обрывается на полуслове")
    if words and DANGLING_VERB.fullmatch(words[-1].lower()):
        problems.append(
            f"заголовок кончается на «{words[-1]}» — не сказано, что именно; "
            f"допиши дополнение или перефразируй")

    # 1а. Подробности-мусор вроде «Ррррр»
    detail = (data.get("detail") or "").strip()
    if detail and (len(detail) < 25 or " " not in detail
                   or re.search(r"(.)\1{3,}", detail)):
        problems.append(
            "раскрывающийся блок бессмысленный — напиши конспект статьи "
            "или оставь его пустым")

    # 1б. Имена латиницей в русском тексте («Дональд Trump»)
    m = LATIN_NAMES.search(ru_text)
    if m:
        problems.append(
            f"имя «{m.group(0)}» написано латиницей — пиши по-русски")

    # 1б'. Выдуманное отчество у иностранца («Иванка Дональдовна Трамп»)
    m = FOREIGN_PATRONYMIC.search(ru_text)
    if m:
        problems.append(
            f"«{m.group(0)}» — у иностранцев нет отчеств, убери его")

    # 1в. Непереведённые английские фразы
    m = ENGLISH_PHRASE.search(f"{data.get('lead', '')} {detail}")
    if m:
        problems.append(
            f"непереведённая английская фраза «{m.group(0)}» — переведи")

    # 2. Имена, которых нет в статье
    for person in data.get("people") or []:
        if isinstance(person, str) and not name_in_source(person, blob):
            problems.append(
                f"имени «{person}» нет в тексте статьи — не придумывай имена")
    for display, pattern, src_forms in KNOWN_PEOPLE:
        if re.search(r"\b" + pattern, ru_text):
            found = any(re.search(r"\b" + re.escape(f), blob) for f in src_forms)
            if not found:
                problems.append(
                    f"в тексте упомянут «{display}», а в статье "
                    f"этого человека нет — бери имена только из статьи")
    return problems


def split_sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?…])\s+", text) if x.strip()]


SYNONYMS = [
    (r"искусственн\w*\s+интеллект\w*", " ии "),
    (r"\bai\b", " ии "),
]


# Общие слова, которые сами по себе ничего не сообщают. Без них
# пересказ «Компания X представила новый сервис на основе ИИ» виден
# как повтор «X запускает ИИ-сервис».
GENERIC = {
    "компа", "предс", "новый", "новая", "новое", "новые", "основ", "работ",
    "котор", "также", "стало", "заяви", "сообщ", "отмет", "рамка", "связи",
    "решен", "напра", "испол", "разра", "проек",
}


def meaning_tokens(text):
    """Значимые слова для сравнения по смыслу: основы по 5 букв,
    синонимы приведены к одному виду, общие слова отброшены."""
    t = text.lower()
    for pat, repl in SYNONYMS:
        t = re.sub(pat, repl, t)
    words = re.findall(r"[a-zа-яё0-9]+", t)
    toks = {w[:5] for w in words if len(w) > 2 and w not in STOPWORDS}
    toks |= {w for w in words if w == "ии"}
    return toks - GENERIC


def clean_detail(data):
    """Убирает из раскрывающегося блока пустые фразы («подробности не
    приводятся», «об этом сообщило агентство») и предложения, которые
    повторяют заголовок и текст под ним. Возвращает сколько убрано."""
    detail = (data.get("detail") or "").strip()
    if not detail:
        return 0
    known = meaning_tokens(f"{data.get('title_ru', '')} {data.get('lead', '')}")
    removed = 0
    new_pars = []
    for par in re.split(r"\n\s*\n", detail):
        kept = []
        for sent in split_sentences(par):
            if FILLER_NEG.search(sent) and FILLER_OBJ.search(sent):
                removed += 1
                continue
            if FILLER_SRC.search(sent) or FILLER_VAGUE.search(sent):
                removed += 1
                continue
            toks = meaning_tokens(sent)
            if toks and len(toks & known) / len(toks) >= 0.6:
                removed += 1
                continue
            kept.append(sent)
            known |= toks
        if kept:
            new_pars.append(" ".join(kept))
    data["detail"] = "\n\n".join(new_pars)
    return removed


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


def normalize_country(raw):
    """Приводит страну к виду «США» или «США/Палестина»: не больше
    двух стран, без повторов, без лишних пробелов вокруг черты."""
    parts = []
    for p in re.split(r"\s*[/|,]\s*|\s+[–—]\s+", str(raw or "")):
        p = p.strip()
        if p and p not in parts:
            parts.append(p)
    return "/".join(parts[:2])


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
    country = normalize_country(data.get("country"))
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

    # Погода, которая держится дольше всего днём. Раньше бралась
    # «худшая» за день: час утреннего тумана превращал солнечный день
    # в «туман». Осадки отдельно сообщает вероятность ниже, а гроза
    # и снег важны, даже если идут недолго.
    code = 0
    if codes:
        code = max(set(codes), key=lambda c: (codes.count(c), c))
        severe = [c for c in codes if c >= 71]
        if severe and code < 71:
            code = max(severe)
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
    started = time.time()
    config = load_config()
    if in_quiet_hours(config):
        print("Тихие часы — новости не отправляю.")
        return

    memory_hours = config.get("duplicate_memory_hours", 48)
    state = load_state()

    # Бота запускает будильник cron-job.org, а расписание GitHub
    # срабатывает редко и в случайное время. Если оно совпадало
    # с будильником, за полчаса приходило 10 новостей вместо 5.
    min_gap = config.get("min_minutes_between_runs", 20)
    last_run = state.get("last_run_ts", 0)
    if (os.environ.get("GITHUB_EVENT_NAME") == "schedule"
            and time.time() - last_run < min_gap * 60):
        print(f"Предыдущий запуск был {int((time.time() - last_run) / 60)} мин "
              f"назад — этот пропускаю, чтобы не было двойной пачки.")
        return

    if not check_telegram():
        print("Останавливаюсь: сначала почините доступ в Telegram (см. выше).")
        return
    state["last_run_ts"] = time.time()

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
    min_common = config.get("duplicate_min_common", 3)
    deadline = started + config.get("max_run_minutes", 8) * 60
    fetch_full = config.get("fetch_full_text", True)
    min_desc = config.get("min_description_chars", 1500)
    attempts = config.get("max_text_attempts", 3)
    kz_share = config.get("kz_share", 0)
    kz_limit = max(1, round(send_limit * kz_share)) if kz_share else 10**9
    patterns = build_keyword_patterns(config.get("keywords", []))
    blocked_list = config.get("blocked_sources", [])
    url_parts = config.get("blocked_url_parts", [])
    per_source = config.get("max_per_source_per_run", 0) or 10**9
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
        if is_blocked(i, blocked_list) or is_blocked_url(i, url_parts):
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
    unique, duplicates = deduplicate(candidates, threshold, state["recent"],
                                     min_common)
    for i in duplicates:
        new_ids.append(i["id"])
    print(f"После склейки дублей осталось: {len(unique)} (убрано {len(duplicates)})")

    # порядок: самые авторитетные источники первыми, но мировые
    # и казахстанские новости вперемешку — в пропорции kz_share
    unique.sort(key=lambda i: -i["trust"])

    # ШАГ 3а — оценка по интересам. Раньше в канал шли первые новости
    # очереди, то есть от изданий с высшим доверием: Gemini одобрял
    # почти всё, а до лент Хизмета и хаджа (доверие 5) очередь не
    # доходила — за 5 дней ни одной такой новости. Теперь один запрос
    # оценивает пул кандидатов, и писать бот будет лучшие.
    ranked = None
    pool_size = config.get("rank_pool_size", 60)
    if pool_size and unique:
        prio = set(config.get("priority_feeds", []))
        bonus_feeds = set(config.get("bonus_feeds", []))
        first = [i for i in unique if i["feed"] in prio]
        rest = [i for i in unique if i["feed"] not in prio]
        pool = (first + build_queue(rest, kz_share))[:pool_size]
        recent_titles = [r["title"] for r in state["recent_ru"][-40:]
                         if r.get("title")]
        scores = rank_candidates(pool, recent_titles, config)
        if scores is None:
            print("[ОЦЕНКА] Gemini не ответил — беру порядок по доверию изданий")
        else:
            min_score = config.get("min_score", 5)
            bonus = config.get("priority_bonus", 2)
            ranked, low, rep = [], 0, 0
            for n, it in enumerate(pool):
                score, repeat = scores.get(n, (min_score, False))
                if repeat:
                    rep += 1
                    new_ids.append(it["id"])
                elif score < min_score:
                    low += 1
                    new_ids.append(it["id"])
                else:
                    it["score"] = score + (bonus if it["feed"] in bonus_feeds else 0)
                    ranked.append(it)
            ranked.sort(key=lambda i: (-i["score"], -i["trust"]))
            print(f"[ОЦЕНКА] оценено {len(pool)}: подходят {len(ranked)}, "
                  f"неинтересных {low}, повторов {rep}")
            for it in ranked[:8]:
                print(f"  {it['score']:>2} · {it['title'][:70]}")

    # Берём с запасом: часть новостей отсеется, если у них не найдётся
    # полного текста статьи.
    batch = build_queue(ranked if ranked is not None else unique,
                        kz_share)[:ai_limit * 2]
    print(f"Кандидатов на обработку: {len(batch)}"
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
    min_text = config.get("min_text_chars", 800)
    ai_used = 0
    last_call = 0.0
    thin = 0
    retried = 0
    rejected = 0
    cleaned = 0
    ai_failed = 0
    models_used = {}
    out_of_time = False
    src_sent = {}
    src_skipped = 0

    def call_ai(it, note=""):
        nonlocal ai_used, last_call
        wait = pause - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()
        ai_used += 1
        data = ask_gemini(it, config, note)
        model = data.pop("_model", None)
        if model:
            models_used[model] = models_used.get(model, 0) + 1
        return data

    def persist():
        """Записывает память на диск сразу после каждой отправки. Если
        GitHub оборвёт запуск по времени, отправленное уже не уйдёт
        повторно."""
        snapshot = dict(state)
        snapshot["seen"] = state["seen"] + new_ids
        save_state(snapshot, memory_hours)

    def unmark(it):
        """Gemini не ответил — новость не считается просмотренной
        и попадёт в следующий запуск."""
        if it["id"] in new_ids:
            new_ids.remove(it["id"])

    for item in batch:
        if sent >= send_limit or ai_used >= ai_limit:
            break
        # Запас до лимита GitHub: одно обращение к Gemini может занять
        # до 3 минут, если все модели тормозят.
        if time.time() > deadline:
            out_of_time = True
            break
        # Gemini не отвечает раз за разом — остальное в следующий запуск
        if ai_failed >= 3:
            break

        group = item["group"]
        best = chosen.get(id(item), item) if fetch_full else item
        if best is None:
            # у всех изданий сюжета настоящий адрес оказался заблокированным
            new_ids.append(item["id"])
            continue

        # Без текста статьи модели не из чего писать: выходит пересказ
        # заголовка на три уровня. Такие новости не присылаем.
        if len(best.get("text") or "") < min_text:
            thin += 1
            new_ids.append(item["id"])
            continue

        if best is not item:
            swapped += 1

        # баланс Казахстан / мир: лимит на казахстанские новости за запуск.
        # Не помечаем как просмотренное — может уйти в следующем запуске.
        if best.get("kz") and kz_sent >= kz_limit:
            kz_skipped += 1
            continue

        # Не больше N новостей одного издания за запуск: за 5 дней
        # Atameken, Guardian и Al Jazeera дали 60% канала.
        # Не помечаем как просмотренное — уйдёт в следующем запуске.
        if src_sent.get(best["source"], 0) >= per_source:
            src_skipped += 1
            continue

        data = call_ai(best)
        new_ids.append(item["id"])
        if not data:
            ai_failed += 1
            unmark(item)
            continue

        if not (data.get("send") and data.get("lead")):
            continue

        # Проверка ответа: оборванный заголовок, выдуманные имена.
        # Одна попытка исправить, потом — не присылаем.
        problems = find_problems(data, best)
        if problems and ai_used < ai_limit and time.time() < deadline:
            retried += 1
            data = call_ai(best, "\n".join(f"- {p}" for p in problems))
            if not data:
                ai_failed += 1
                unmark(item)
                continue
            if not (data.get("send") and data.get("lead")):
                continue
            problems = find_problems(data, best)
        if problems:
            rejected += 1
            print(f"  отклонено после проверки: «{data.get('title_ru','')[:50]}»"
                  f" — {problems[0]}")
            continue

        cleaned += clean_detail(data)

        is_kz = best.get("kz") or "Казахстан" in normalize_country(
            data.get("country")).split("/")
        if is_kz and kz_sent >= kz_limit:
            # новость из мировой ленты оказалась о Казахстане — отложить
            # до следующего запуска, а не терять
            kz_skipped += 1
            unmark(item)
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
            src_sent[best["source"]] = src_sent.get(best["source"], 0) + 1
            state["recent"].append(
                {"tokens": sorted(item["tokens"]), "ts": time.time()}
            )
            # заголовок нужен оценке: она сверяет новых кандидатов
            # с уже отправленным
            state["recent_ru"].append(
                {"tokens": sorted(ru_tokens), "ts": time.time(),
                 "title": data.get("title_ru", "")}
            )
            persist()

    state["seen"] = state["seen"] + new_ids
    save_state(state, memory_hours)
    if out_of_time:
        print("Время запуска вышло — остальные новости в следующем запуске.")
    if ai_failed >= 3:
        print("Gemini не отвечает — остальные новости в следующем запуске.")
    models_txt = ", ".join(f"{m} {n}" for m, n in models_used.items()) or "нет"
    print(
        f"Обращений к Gemini: {ai_used} (повторных: {retried}, "
        f"без ответа: {ai_failed}). Ответили модели: {models_txt}. "
        f"Догружено статей: {enriched}. "
        f"Без текста статьи, пропущено: {thin}. "
        f"Отклонено проверкой: {rejected}. "
        f"Убрано пустых и повторных фраз: {cleaned}. "
        f"Текст взят у другого издания сюжета: {swapped}. "
        f"Отсеяно как дубль после перевода: {ru_dups}. "
        f"Отложено по балансу Казахстан/мир: {kz_skipped}. "
        f"Отложено по лимиту на издание: {src_skipped}. "
        f"Отправлено сообщений: {sent} (из них о Казахстане: {kz_sent})"
    )


if __name__ == "__main__":
    main()
