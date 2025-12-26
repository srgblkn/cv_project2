from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Torch импортируем лениво (только при инференсе), чтобы страница с логами не падала
# если вдруг torch не установлен/конфликт версий.


# -----------------------------
# Paths (строго)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "forrest"

MODEL_PY = ART_DIR / "model_class.py"
LOG_CSV = ART_DIR / "unet_training_log.csv"
BG_JPG = ART_DIR / "forrest.jpg"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Forest Segmentation", page_icon="🌲", layout="wide")


# -----------------------------
# UI: background + opaque cards
# -----------------------------
def apply_background(bg_path: Path) -> None:
    bg_css = ""
    if bg_path.exists():
        b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
        bg_css = f"""
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    st.markdown(
        f"""
        <style>
        {bg_css}

        .stApp, .stMarkdown, .stText, .stCaption, .stWrite {{
            color: #F8FAFC;
        }}

        header[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

        section[data-testid="stSidebar"] {{
            background: #0B1220;
            border-right: 1px solid rgba(255,255,255,0.10);
        }}
        section[data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}

        .opaque-card {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.40);
            margin-bottom: 14px;
        }}
        .opaque-card h3 {{
            margin: 0;
            font-size: 1.25rem;
            font-weight: 750;
            color: #F8FAFC;
        }}
        .opaque-card p {{
            margin: 6px 0 0 0;
            color: rgba(248,250,252,0.85);
            line-height: 1.35;
        }}

        div[data-testid="stExpander"] > details {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 10px 12px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.30);
        }}
        div[data-testid="stExpander"] summary {{
            color: #F8FAFC !important;
            font-weight: 650;
        }}

        div[data-testid="stFileUploader"] section {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 10px;
        }}

        .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.14);
        }}

        a {{ color: #93C5FD !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def opaque_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="opaque-card">
          <h3>{title}</h3>
          <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_switch_page(target: str) -> None:
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            st.info("Переход недоступен. Используйте меню слева.")
    else:
        st.info("Переход недоступен. Используйте меню слева.")


apply_background(BG_JPG)


# -----------------------------
# Model import (from existing model_class.py)
# -----------------------------
def import_unet_class(model_py: Path):
    if not model_py.exists():
        raise FileNotFoundError(f"Не найден файл модели: {model_py.as_posix()}")

    spec = importlib.util.spec_from_file_location("forrest_model_class", model_py.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось создать import spec для model_class.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # может бросить ImportError (например, torchmetrics)
    if not hasattr(module, "UNet"):
        raise AttributeError("В model_class.py не найден класс UNet")
    return module.UNet


def find_weight_candidates(dir_path: Path) -> list[Path]:
    exts = ("*.pt", "*.pth", "*.ckpt")
    files: list[Path] = []
    for pat in exts:
        files.extend(dir_path.glob(pat))
    # сортировка: более “похожее на best” вверх
    files = sorted(files, key=lambda p: ("best" not in p.name.lower(), p.name.lower()))
    return files


# -----------------------------
# Image utils
# -----------------------------
def to_tensor_rgb(img: Image.Image):
    # lazy torch import
    import torch

    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    return t


def sigmoid_mask(logits):
    import torch

    probs = torch.sigmoid(logits)
    return probs


def overlay_mask_on_image(img: Image.Image, mask_2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    mask_2d: float/uint8 [H,W] where 1=forest, 0=background
    """
    base = img.convert("RGBA")
    h, w = mask_2d.shape
    # зелёная заливка
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay_np = np.array(overlay)

    m = (mask_2d > 0.5)
    overlay_np[m] = np.array([46, 204, 113, int(255 * alpha)], dtype=np.uint8)  # green
    overlay = Image.fromarray(overlay_np, mode="RGBA")

    return Image.alpha_composite(base, overlay).convert("RGB")


def mask_to_png_bytes(mask_2d: np.ndarray) -> bytes:
    # маска как 0/255
    m = (mask_2d > 0.5).astype(np.uint8) * 255
    im = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------
# Header
# -----------------------------
opaque_card(
    "Forest Segmentation",
    "Автоматическая маска покрытия по спутниковым снимкам: превью, наложение и выгрузка результата.",
)

h1, h2 = st.columns([1, 1], gap="large")
with h1:
    if st.button("← На главную", use_container_width=True):
        safe_switch_page("app.py")
with h2:
    if BG_JPG.exists():
        st.download_button(
            "Скачать фон (JPG)",
            data=BG_JPG.read_bytes(),
            file_name=BG_JPG.name,
            mime="image/jpeg",
            use_container_width=True,
        )


# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.markdown("## Настройки")
preset = st.sidebar.selectbox("Режим обработки", ["Fast", "Balanced", "High"], index=1)
size_map = {"Fast": 256, "Balanced": 384, "High": 512}
img_size = size_map[preset]

threshold = st.sidebar.slider("Порог маски", 0.05, 0.95, 0.50, 0.05)
alpha = st.sidebar.slider("Прозрачность наложения", 0.10, 0.90, 0.45, 0.05)

export_mode = st.sidebar.selectbox("Экспорт", ["ZIP (маски + overlay)", "ZIP (только маски)"], index=0)


# -----------------------------
# Training log (unet_training_log.csv): charts by request
# -----------------------------
log_df = None
if LOG_CSV.exists():
    try:
        log_df = pd.read_csv(LOG_CSV)
    except Exception:
        log_df = None

right_info, right_charts = st.columns([1.0, 1.0], gap="large")

with right_info:
    opaque_card("Параметры обучения", "Ключевые метрики и динамика обучения (по лог-файлу).")
    if log_df is None:
        st.info(f"Лог не найден или не читается: `{LOG_CSV.as_posix()}`")
    else:
        # аккуратно ищем epoch
        epoch_col = next((c for c in log_df.columns if c.lower() in ("epoch", "epochs", "step")), None)
        # компактная сводка “последняя строка”
        tail = log_df.tail(1).copy()
        st.dataframe(tail, use_container_width=True, hide_index=True)

with right_charts:
    opaque_card("Графики обучения", "Выберите показатели для отображения.")
    if log_df is None:
        st.info("Нет данных для построения графиков.")
    else:
        epoch_col = next((c for c in log_df.columns if c.lower() in ("epoch", "step")), None)
        if epoch_col is None:
            st.dataframe(log_df.tail(30), use_container_width=True)
        else:
            numeric_cols = [c for c in log_df.columns if c != epoch_col and pd.api.types.is_numeric_dtype(log_df[c])]
            if not numeric_cols:
                st.dataframe(log_df.tail(30), use_container_width=True)
            else:
                default = numeric_cols[:1]
                selected = st.multiselect("Показатели", options=numeric_cols, default=default)
                if selected:
                    chart = log_df[[epoch_col] + selected].copy().set_index(epoch_col)
                    st.line_chart(chart, use_container_width=True)


st.divider()


# -----------------------------
# Inference UI
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    opaque_card("Загрузка снимков", "Загрузите один или несколько файлов. Результат можно скачать одним ZIP.")
    uploads = st.file_uploader(
        "Images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploads:
        with st.expander("Предпросмотр", expanded=True):
            cols = st.columns(4)
            for i, up in enumerate(uploads):
                try:
                    up.seek(0)
                    img = Image.open(up).convert("RGB")
                    cols[i % 4].image(img, caption=up.name, use_container_width=True)
                    up.seek(0)
                except Exception:
                    cols[i % 4].write(up.name)

    run_btn = st.button("Запустить сегментацию", type="primary", use_container_width=True)

with right:
    opaque_card("Веса модели", "Автопоиск весов в папке `pages/forrest/` или загрузка вручную.")
    candidates = find_weight_candidates(ART_DIR)
    chosen_path = None

    if candidates:
        chosen_name = st.selectbox("Найденные веса", [p.name for p in candidates], index=0)
        chosen_path = ART_DIR / chosen_name
        st.caption(f"Будет использовано: `{chosen_path.as_posix()}`")
    else:
        st.warning("Файлы весов (*.pt/*.pth/*.ckpt) рядом не найдены.")
        uploaded_weights = st.file_uploader("Загрузить веса", type=["pt", "pth", "ckpt"], accept_multiple_files=False)
        if uploaded_weights is not None:
            # сохраняем во временный файл (внутри контейнера)
            tmp_path = Path("/tmp") / uploaded_weights.name
            tmp_path.write_bytes(uploaded_weights.getbuffer())
            chosen_path = tmp_path
            st.caption(f"Будет использовано: `{chosen_path.as_posix()}`")

    st.caption(f"Модель-класс: `{MODEL_PY.as_posix()}`")
    st.caption(f"Лог: `{LOG_CSV.as_posix()}`")
    st.caption(f"Фон: `{BG_JPG.as_posix()}`")


# -----------------------------
# Run inference
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_weights(model_py: str, weights_path: str):
    import torch

    UNet = import_unet_class(Path(model_py))  # может упасть по ImportError (torchmetrics и т.п.)
    model = UNet(n_class=1)

    ckpt = torch.load(weights_path, map_location="cpu")

    # распространённые форматы чекпойнтов
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    else:
        # очень редкий случай: сохранён целиком model
        state = None

    if state is not None:
        # иногда ключи бывают с префиксом "module."
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)

    model.eval()
    return model


if run_btn:
    if not uploads:
        st.warning("Загрузите хотя бы один файл.")
        st.stop()
    if chosen_path is None:
        st.error("Не выбраны веса модели. Положите файл весов в `pages/forrest/` или загрузите вручную.")
        st.stop()

    try:
        with st.spinner("Загружаю модель и веса..."):
            model = load_model_and_weights(MODEL_PY.as_posix(), chosen_path.as_posix())
    except ImportError as e:
        st.error(
            "Не удалось импортировать `model_class.py` из-за отсутствующей зависимости.\n\n"
            f"Ошибка: {e}\n\n"
            "Решение: добавьте недостающий пакет в `requirements.txt` (например, `torchmetrics`)."
        )
        st.stop()
    except Exception as e:
        st.error(f"Ошибка загрузки модели/весов: {e}")
        st.stop()

    import torch

    results_for_zip: list[tuple[str, bytes]]
