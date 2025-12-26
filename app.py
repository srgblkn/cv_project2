# app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st


APP_TITLE = "Vision Suite"
APP_SUBTITLE = "Компьютерное зрение для прикладных бизнес-сценариев"

PAGES_DIR = Path("pages")

FACE_PAGE = PAGES_DIR / "facescanner.py"
CANCER_PAGE = PAGES_DIR / "cancer.py"

HOME_SCRIPT = "app.py"


def _as_streamlit_page_path(page_path: Path | str) -> str:
    """
    Streamlit ожидает путь к странице в виде строки, как в файловой структуре репозитория.
    Важно: используем forward slashes.
    """
    p = Path(page_path) if isinstance(page_path, str) else page_path
    return p.as_posix()


def nav_button(page_path: Path | str, label: str, icon: str | None = None, *, location: str = "main"):
    """
    Надёжная навигация без st.page_link.
    - Если страница существует и доступен st.switch_page -> кликабельная кнопка.
    - Иначе -> disabled + пояснение.
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
        if not exists:
            container.caption(
                f"Страница не найдена: `{p.as_posix()}`. Проверьте, что файл закоммичен и лежит в папке `pages/`."
            )
        elif not has_switch:
            container.caption("Навигация недоступна в текущей версии Streamlit. Используйте меню слева.")


def render_header():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <div style="padding: 0.2rem 0 0.6rem 0;">
          <div style="font-size: 2.2rem; font-weight: 700; line-height: 1.1;">{APP_TITLE}</div>
          <div style="font-size: 1.05rem; opacity: 0.85; margin-top: 0.35rem;">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### Навигация")
    st.sidebar.caption("Выберите модуль.")

    st.sidebar.markdown("#### Быстрый старт")
    nav_button(FACE_PAGE, "FaceScanner — маскировка лиц", "🕵️", location="sidebar")
    nav_button(CANCER_PAGE, "BrainScan Detect — анализ снимков", "🧠", location="sidebar")

    st.sidebar.divider()
    st.sidebar.markdown("#### О платформе")
    st.sidebar.write("• Пакетная обработка файлов")
    st.sidebar.write("• Быстрый прототип → замена весов без изменения UI")
    st.sidebar.write("• Понятный формат результатов (превью + скачивание)")

    st.sidebar.divider()
    st.sidebar.caption(f"Сессия: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def render_hero():
    c1, c2 = st.columns([1.25, 1.0], gap="large")
    with c1:
        st.markdown("### Три решения — три сценария")
        st.write(
            "Набор модулей компьютерного зрения для задач, где важны скорость, понятный результат и удобная упаковка "
            "в интерфейс для конечного пользователя."
        )
    with c2:
        st.markdown("### Открыть модуль")
        nav_button(FACE_PAGE, "Открыть FaceScanner", "🕵️", location="main")


def render_solution_cards():
    st.markdown("### Решения")
    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown("#### 1) FaceScanner")
        st.caption("Анонимизация изображений")
        st.write("Детекция лиц и маскировка области. Поддерживает загрузку нескольких файлов.")
        nav_button(FACE_PAGE, "Перейти", "🕵️", location="main")

    with b:
        st.markdown("#### 2) BrainScan Detect")
        st.caption("Анализ снимков")
        st.write("Модуль анализа снимков: пакетная загрузка, превью и экспорт результатов.")
        nav_button(CANCER_PAGE, "Перейти", "🧠", location="main")

    with c:
        st.markdown("#### 3) Forest Segmentation")
        st.caption("Сегментация лесных массивов")
        st.write("Семантическая сегментация спутниковых снимков (бинарные маски) для мониторинга покрытий и изменений.")
        st.button("Скоро доступно", use_container_width=True, disabled=True)


def render_flow():
    st.markdown("### Как пользоваться")
    x1, x2, x3 = st.columns(3, gap="large")

    with x1:
        st.markdown("**1. Загрузка**")
        st.write("Загрузите один или несколько файлов. В некоторых модулях будет доступна загрузка по ссылке.")

    with x2:
        st.markdown("**2. Обработка**")
        st.write("Модель выполняет инференс. Настройки позволяют адаптировать строгость под задачу.")

    with x3:
        st.markdown("**3. Результат**")
        st.write("Просмотр превью и скачивание итогов (например, ZIP с обработанными файлами).")


def render_footer():
    st.divider()
    st.caption("Работу выполнили студенты Эльбруса — Игорь Никовский и Сергей Белькин")


def main():
    render_header()
    render_sidebar()

    render_hero()
    st.divider()

    render_solution_cards()
    st.divider()

    render_flow()
    render_footer()


if __name__ == "__main__":
    main()
