from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import streamlit as st


# =============================
# Константы
# =============================
APP_TITLE = "Vision Suite"
APP_SUBTITLE = "Компьютерное зрение для прикладных бизнес-сценариев"

PAGES_DIR = Path("pages")
FACE_PAGE = PAGES_DIR / "facescanner.py"
CANCER_PAGE = PAGES_DIR / "cancer.py"
FORREST_PAGE = PAGES_DIR / "forrest.py"  # важно: именно forrest.py (как у вас в проекте)

BG_PATH = Path("screen.jpg")  # фон лежит в корне


# =============================
# Утилиты
# =============================
def _as_streamlit_page_path(page_path: Path | str) -> str:
    p = Path(page_path) if isinstance(page_path, str) else page_path
    return p.as_posix()


def _apply_background(bg_path: Path) -> None:
    if bg_path.exists():
        b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
        bg_css = (
            '.stApp{'
            f'background-image:url("data:image/jpeg;base64,{b64}");'
            "background-size:cover;"
            "background-position:center;"
            "background-attachment:fixed;"
            "}"
        )
    else:
        bg_css = ""

    st.markdown(
        f"""
<style>
{bg_css}

.stApp, .stMarkdown, .stText, .stCaption, .stWrite {{
  color:#F8FAFC;
}}
header[data-testid="stHeader"] {{
  background: rgba(0,0,0,0);
}}

section[data-testid="stSidebar"] {{
  background:#0B1220;
  border-right:1px solid rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] * {{
  color:#F8FAFC !important;
}}

.opaque-card {{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:16px 16px 14px 16px;
  box-shadow:0 10px 24px rgba(0,0,0,0.40);
  margin-bottom:14px;
  text-align:center;
}}
.opaque-card * {{ text-align:center; }}

.opaque-card h1 {{
  margin:0;
  font-size:2.05rem;
  font-weight:780;
  line-height:1.15;
}}
.opaque-card h3 {{
  margin:0;
  font-size:1.20rem;
  font-weight:750;
}}
.opaque-card p {{
  margin:8px 0 0 0;
  color:rgba(248,250,252,0.85);
  line-height:1.35;
}}

.stButton > button {{
  border-radius:14px;
  border:1px solid rgba(255,255,255,0.14);
}}
a {{ color:#93C5FD !important; }}

.solution-card {{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:16px;
  box-shadow:0 10px 24px rgba(0,0,0,0.30);
  height: 100%;
}}
.solution-title {{
  font-size:1.10rem;
  font-weight:780;
  margin:0 0 6px 0;
}}
.solution-sub {{
  color:rgba(248,250,252,0.75);
  margin:0 0 10px 0;
}}
.solution-desc {{
  color:rgba(248,250,252,0.88);
  margin:0 0 14px 0;
  line-height:1.35;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _title_card(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="opaque-card"><h1>{title}</h1>{subtitle_html}</div>', unsafe_allow_html=True)


def _card(title: str, text: str | None = None) -> None:
    text = text or ""
    st.markdown(f'<div class="opaque-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)


def nav_button(page_path: Path | str, label: str, icon: str | None = None, *, location: str = "main") -> None:
    """
    Надёжная навигация без st.page_link.
    Если файла нет/нет switch_page — кнопка будет отключена (без тех. сообщений на главной).
    """
    p = Path(page_path) if isinstance(page_path, str) else page_path
    text = f"{icon} {label}" if icon else label

    exists = p.exists()
    has_switch = hasattr(st, "switch_page")
    container = st.sidebar if location == "sidebar" else st

    if exists and has_switch:
        if container.button(text, use_container_width=True):
            st.switch_page(_as_streamlit_page_path(p))
    else:
        container.button(text, use_container_width=True, disabled=True)


# =============================
# Рендер
# =============================
def render_header() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_background(BG_PATH)
    _title_card(APP_TITLE, APP_SUBTITLE)


def render_sidebar() -> None:
    st.sidebar.markdown("### Модули")
    nav_button(FACE_PAGE, "FaceScanner — маскировка лиц", "🕵️", location="sidebar")
    nav_button(CANCER_PAGE, "BrainScan Detect — анализ снимков", "🧠", location="sidebar")
    nav_button(FORREST_PAGE, "Сегментация леса на аэрокосмических снимках", "🌲", location="sidebar")

    st.sidebar.divider()
    st.sidebar.markdown("### Сессия")
    st.sidebar.caption(datetime.now().strftime("%Y-%m-%d %H:%M"))


def render_hero() -> None:
    _card(
        "Три решения — три сценария",
        "Набор инструментов компьютерного зрения для задач, где важны скорость, понятный результат и удобная упаковка "
        "в интерфейс для конечного пользователя.",
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        nav_button(FACE_PAGE, "Открыть FaceScanner", "🕵️", location="main")
    with c2:
        nav_button(CANCER_PAGE, "Открыть BrainScan Detect", "🧠", location="main")
    with c3:
        nav_button(FORREST_PAGE, "Открыть сегментацию леса", "🌲", location="main")


def render_solution_cards() -> None:
    st.markdown('<div class="opaque-card"><h3>Решения</h3></div>', unsafe_allow_html=True)

    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown(
            """
<div class="solution-card">
  <div class="solution-title">FaceScanner</div>
  <div class="solution-sub">Анонимизация изображений</div>
  <div class="solution-desc">Детекция лиц и маскировка области. Пакетная загрузка, предпросмотр и экспорт результатов.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        nav_button(FACE_PAGE, "Перейти", "🕵️", location="main")

    with b:
        st.markdown(
            """
<div class="solution-card">
  <div class="solution-title">BrainScan Detect</div>
  <div class="solution-sub">Анализ снимков</div>
  <div class="solution-desc">Пакетная обработка изображений, локализация зон интереса и экспорт результатов одним архивом.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        nav_button(CANCER_PAGE, "Перейти", "🧠", location="main")

    with c:
        st.markdown(
            """
<div class="solution-card">
  <div class="solution-title">Сегментация леса</div>
  <div class="solution-sub">Аэрокосмические снимки</div>
  <div class="solution-desc">Семантическая сегментация: выделение лесных массивов маской на аэрокосмических снимках.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        # ВАЖНО: здесь теперь реальная ссылка/кнопка на pages/forrest.py
        nav_button(FORREST_PAGE, "Перейти", "🌲", location="main")


def render_flow() -> None:
    st.markdown('<div class="opaque-card"><h3>Как пользоваться</h3></div>', unsafe_allow_html=True)

    x1, x2, x3 = st.columns(3, gap="large")

    with x1:
        _card("1. Загрузка", "Загрузите один или несколько файлов. В некоторых модулях доступна загрузка по ссылке.")
    with x2:
        _card("2. Обработка", "Модель выполняет инференс. Настройки позволяют адаптировать строгость под задачу.")
    with x3:
        _card("3. Результат", "Просмотр превью и скачивание итогов (например, ZIP с обработанными файлами).")


def render_footer() -> None:
    st.divider()
    st.markdown(
        '<div class="opaque-card"><p>Работу выполнили студенты Эльбруса — Игорь Никовский и Сергей Белькин</p></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()
    render_sidebar()

    render_hero()
    render_solution_cards()
    render_flow()
    render_footer()


if __name__ == "__main__":
    main()
