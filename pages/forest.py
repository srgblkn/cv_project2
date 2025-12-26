import base64
import io
import zipfile
import inspect
from pathlib import Path
import importlib.util
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image


# -----------------------------
# Пути (строго под вашу структуру)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "forrest"

BG_JPG = ART_DIR / "forrest.jpg"
MODEL_PY = ART_DIR / "model_class.py"
WEIGHTS_PTH = ART_DIR / "model_unet.pth"
TRAIN_LOG = ART_DIR / "unet_training_log.csv"


# -----------------------------
# Страница
# -----------------------------
st.set_page_config(page_title="Сегментация аэрокосмических снимков", page_icon="*", layout="wide")


# -----------------------------
# Дизайн
# -----------------------------
UPLOAD_BOX_H = 90   # компактно (в ~3 раза меньше исходного)
CHART_H = 340       # одинаковая высота графика и "лучшие метрики"


def apply_background_and_theme(bg_path: Path) -> None:
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

        .stApp, .stMarkdown, .stText, .stCaption, .stWrite {{ color: #F8FAFC; }}
        header[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

        section[data-testid="stSidebar"] {{
            background: #0B1220;
            border-right: 1px solid rgba(255,255,255,0.10);
        }}
        section[data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}

        /* Подложка: ВСЕ тексты по центру */
        .opaque-card {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.40);
            margin-bottom: 14px;
            text-align: center;
        }}
        .opaque-card * {{ text-align: center; }}

        .opaque-card h1 {{
            margin: 0;
            font-size: 2.0rem;
            font-weight: 780;
            line-height: 1.15;
        }}
        .opaque-card h3 {{
            margin: 0;
            font-size: 1.25rem;
            font-weight: 750;
        }}
        .opaque-card p {{
            margin: 8px 0 0 0;
            color: rgba(248,250,252,0.85);
            line-height: 1.35;
        }}

        /* Экспандеры тоже на подложке */
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

        /* ЖЁСТКО фиксируем высоту аплоадеров/поля */
        div[data-testid="stFileUploader"] section {{
            height: {UPLOAD_BOX_H}px !important;
            overflow: auto !important;
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 10px;
        }}
        div[data-testid="stTextArea"] textarea {{
            height: {UPLOAD_BOX_H}px !important;
        }}
        div[data-testid="stTextArea"] > div {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 10px;
        }}

        .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.14);
        }}

        /* Карточка метрик фиксированной высоты + вертикальное центрирование */
        .metrics-card {{
            height: {CHART_H}px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 12px;
        }}
        .metric-line {{
            line-height: 1.2;
        }}
        .muted {{
            color: rgba(248,250,252,0.70);
            font-size: 0.95rem;
        }}
        .metric-value {{
            font-size: 1.55rem;
            font-weight: 780;
            margin-top: 4px;
        }}

        /* "Табличные" ячейки в строку (без границ) */
        .kpi-row {{
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 10px;
        }}
        .kpi-cell {{
            min-width: 140px;
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.06);
        }}
        .kpi-title {{
            color: rgba(248,250,252,0.70);
            font-size: 0.92rem;
            margin-bottom: 4px;
        }}
        .kpi-val {{
            font-weight: 780;
            font-size: 1.05rem;
            color: rgba(248,250,252,0.95);
        }}

        a {{ color: #93C5FD !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def title_card(title: str) -> None:
    st.markdown(f'<div class="opaque-card"><h1>{title}</h1></div>', unsafe_allow_html=True)


def card(title: str, text: str | None = None) -> None:
    text = text or ""
    st.markdown(f'<div class="opaque-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)


def card_html(html_inner: str) -> None:
    # Передавайте сюда только “внутренности” карточки без начальных отступов в строках
    st.markdown(f'<div class="opaque-card">{html_inner}</div>', unsafe_allow_html=True)


def safe_switch_page(target: str) -> None:
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            pass


apply_background_and_theme(BG_JPG)


# -----------------------------
# Sidebar
# -----------------------------
if st.sidebar.button("На главную", use_container_width=True):
    safe_switch_page("app.py")

st.sidebar.divider()
st.sidebar.markdown("## Настройки инференса")
preset = st.sidebar.selectbox("Режим", ["Быстро", "Сбалансировано", "Максимум"], index=1)
size_map = {"Быстро": 256, "Сбалансировано": 384, "Максимум": 512}
img_size = int(size_map[preset])

threshold = st.sidebar.slider("Порог маски", 0.05, 0.95, 0.50, 0.05)
alpha = st.sidebar.slider("Прозрачность наложения", 0.10, 0.90, 0.45, 0.05)
export_mode = st.sidebar.selectbox("Экспорт", ["ZIP (маски + наложение)", "ZIP (только маски)"], index=0)


# -----------------------------
# Загрузка изображений
# -----------------------------
def _download_url_bytes(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _load_images_from_uploads(files) -> list[tuple[str, Image.Image]]:
    out = []
    if not files:
        return out
    for f in files:
        try:
            f.seek(0)
            img = Image.open(f).convert("RGB")
            f.seek(0)
            out.append((f.name, img))
        except Exception:
            continue
    return out


def _load_images_from_urls(urls: list[str]) -> list[tuple[str, Image.Image]]:
    out = []
    for u in urls:
        try:
            b = _download_url_bytes(u)
            img = Image.open(io.BytesIO(b)).convert("RGB")
            name = Path(u.split("?")[0]).name or "image.jpg"
            out.append((name, img))
        except Exception:
            continue
    return out


# -----------------------------
# Импорт модели + веса
# -----------------------------
def import_model_module(model_py: Path):
    spec = importlib.util.spec_from_file_location("forrest_model_class", model_py.as_posix())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def pick_model_class(module):
    import torch.nn as nn

    if hasattr(module, "UNet"):
        cls = getattr(module, "UNet")
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return cls, "UNet"

    candidates = []
    for name, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            if getattr(obj, "__module__", "") == module.__name__:
                candidates.append((name, obj))
    candidates.sort(key=lambda x: x[0].lower())
    return candidates[0][1], candidates[0][0]


def build_model(model_cls):
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.values())[1:]  # без self
    kwargs = {}

    name_map = {
        "n_class": 1,
        "n_classes": 1,
        "num_classes": 1,
        "classes": 1,
        "out_channels": 1,
        "n_channels": 3,
        "in_channels": 3,
        "channels": 3,
    }
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in name_map:
            kwargs[p.name] = name_map[p.name]

    try:
        return model_cls(**kwargs)
    except Exception:
        return model_cls()


def load_weights_into_model(model, weights_path: Path):
    import torch

    ckpt = torch.load(weights_path.as_posix(), map_location="cpu")
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        return ckpt

    if isinstance(ckpt, dict):
        state = ckpt["state_dict"] if ("state_dict" in ckpt and isinstance(ckpt["state_dict"], dict)) else ckpt
        cleaned = {k.replace("module.", ""): v for k, v in state.items() if isinstance(k, str)}
        model.load_state_dict(cleaned, strict=False)
        return model

    return model


@st.cache_resource(show_spinner=False)
def load_model_cached(model_py: str, weights_path: str):
    import torch  # noqa: F401

    module = import_model_module(Path(model_py))
    model_cls, model_name = pick_model_class(module)
    model = build_model(model_cls)
    model = load_weights_into_model(model, Path(weights_path))
    model.eval()
    return model, model_name


# -----------------------------
# Сегментация
# -----------------------------
def to_tensor_rgb(img: Image.Image):
    import torch

    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return t


def logits_to_prob_binary(y):
    import torch

    if isinstance(y, (list, tuple)):
        y = y[0]

    if hasattr(y, "dim") and y.dim() == 4:
        if y.size(1) == 1:
            return torch.sigmoid(y)[0, 0]
        if y.size(1) == 2:
            return torch.softmax(y, dim=1)[0, 1]
        return torch.softmax(y, dim=1)[0, 0]

    if hasattr(y, "dim") and y.dim() == 3:
        return torch.sigmoid(y)[0]

    raise RuntimeError("Неподдерживаемая форма выхода модели.")


def overlay_mask(img_rgb: Image.Image, mask01: np.ndarray, alpha_: float):
    base = img_rgb.convert("RGBA")
    h, w = mask01.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov = np.array(overlay)
    m = mask01 > 0.5
    ov[m] = np.array([46, 204, 113, int(255 * alpha_)], dtype=np.uint8)
    overlay = Image.fromarray(ov, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def mask_png_bytes(mask01: np.ndarray) -> bytes:
    m = (mask01 > 0.5).astype(np.uint8) * 255
    im = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def img_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------
# Лог обучения
# -----------------------------
def load_training_log(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception:
        return None


def pick_epoch_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).lower() in ("epoch", "step"):
            return c
    return None


def find_metric_col(df: pd.DataFrame, substrs: list[str]) -> str | None:
    cols_lower = {c: str(c).lower() for c in df.columns}
    for c, cl in cols_lower.items():
        if any(s in cl for s in substrs):
            return c
    return None


# -----------------------------
# UI layout
# -----------------------------
title_card("Сегментация аэрокосмических снимков")

card("Загрузка изображений", "Загрузите изображения файлами и/или добавьте прямые ссылки")

u_cols = st.columns([1, 1], gap="large")
with u_cols[0]:
    card("Загрузка файлами", "")
    img_files = st.file_uploader(
        "Файлами",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with u_cols[1]:
    card("Загрузка по ссылкам", "")
    img_urls_text = st.text_area(
        "По ссылкам",
        height=UPLOAD_BOX_H,
        placeholder="https://...\nhttps://...",
        label_visibility="collapsed",
    )

images = []
images.extend(_load_images_from_uploads(img_files))
images.extend(_load_images_from_urls(_urls_from_text(img_urls_text)))

if images:
    with st.expander("Предпросмотр загруженных изображений", expanded=True):
        cols = st.columns(4)
        for i, (name, img) in enumerate(images):
            cols[i % 4].image(img, caption=name, use_container_width=True)

run_btn = st.button("Запустить сегментацию", type="primary", use_container_width=True)

if run_btn and images:
    import torch

    with st.spinner("Выполняется сегментация..."):
        model, _model_name = load_model_cached(MODEL_PY.as_posix(), WEIGHTS_PTH.as_posix())

    results_for_zip = []
    preview_rows = []

    prog = st.progress(0)
    for idx, (name, orig) in enumerate(images, start=1):
        resized = orig.resize((img_size, img_size), resample=Image.BILINEAR)
        x = to_tensor_rgb(resized)

        with torch.no_grad():
            y = model(x)

        prob_small = logits_to_prob_binary(y).detach().cpu().numpy()
        mask_small = (prob_small > float(threshold)).astype(np.float32)

        mask_img = Image.fromarray((mask_small * 255).astype(np.uint8), mode="L").resize(
            orig.size, resample=Image.NEAREST
        )
        mask01_orig = (np.array(mask_img).astype(np.float32) / 255.0)
        overlay = overlay_mask(orig, mask01_orig, alpha_=float(alpha))

        stem = Path(name).stem
        if export_mode == "ZIP (маски + наложение)":
            results_for_zip.append((f"{stem}_маска.png", mask_png_bytes(mask01_orig)))
            results_for_zip.append((f"{stem}_наложение.png", img_png_bytes(overlay)))
        else:
            results_for_zip.append((f"{stem}_маска.png", mask_png_bytes(mask01_orig)))

        coverage = float(mask01_orig.mean())
        preview_rows.append((name, orig, overlay, coverage))
        prog.progress(int(idx / len(images) * 100))
    prog.empty()

    card("Результаты", "Превью и выгрузка результатов одним архивом.")
    for name, orig, overlay, cov in preview_rows:
        with st.expander(f"{name} — доля маски: {cov * 100:.1f}%", expanded=False):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                card("Оригинал", "")
                st.image(orig, use_container_width=True)
            with c2:
                card("Наложение маски", "")
                st.image(overlay, use_container_width=True)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        "Скачать архив с результатами",
        data=zip_buf,
        file_name="результаты_сегментации.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.divider()

# -----------------------------
# Качество: 1 подложка (заголовок + выбор показателей)
# -----------------------------
log_df = load_training_log(TRAIN_LOG)
epoch_col = pick_epoch_col(log_df) if log_df is not None else None

selected = []
if log_df is not None and epoch_col is not None:
    numeric_cols = [
        c for c in log_df.columns
        if c != epoch_col and pd.api.types.is_numeric_dtype(log_df[c])
    ]
else:
    numeric_cols = []

# Одна подложка: заголовок + селект
card_html(
    '<h3>Показатели качества, рассчитаные на валидационной выборке</h3>'
    '<p style="margin-top:10px;margin-bottom:0;">Выберите показатели, которые нужно отобразить</p>'
)

if numeric_cols:
    selected = st.multiselect("Показатели", options=numeric_cols, default=numeric_cols[:3], label_visibility="collapsed")

# -----------------------------
# Подложка на всю ширину: "Интерактивная визуализация..."
# Ниже: графики слева, лучшие метрики справа
# -----------------------------
card("Интерактивная визуализация динамики метрик", "")

g_col, m_col = st.columns([1.35, 0.65], gap="large")

with g_col:
    if log_df is None or epoch_col is None or not selected:
        with st.expander("Таблица (фрагмент)", expanded=True):
            if log_df is not None:
                st.dataframe(log_df.tail(30), use_container_width=True)
            else:
                st.write("—")
    else:
        long = log_df[[epoch_col] + selected].melt(id_vars=[epoch_col], var_name="Показатель", value_name="Значение")
        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X(f"{epoch_col}:Q", title="Эпоха"),
                y=alt.Y("Значение:Q", title="Значение"),
                color=alt.Color("Показатель:N", title=""),
                tooltip=[
                    alt.Tooltip(f"{epoch_col}:Q", title="Эпоха"),
                    alt.Tooltip("Показатель:N", title="Показатель"),
                    alt.Tooltip("Значение:Q", title="Значение", format=".6f"),
                ],
            )
            .interactive()
            .properties(height=CHART_H)
        )
        st.altair_chart(chart, use_container_width=True)

with m_col:
    if log_df is None:
        st.markdown(f'<div class="opaque-card metrics-card"><h3>Лучшие метрики</h3><div class="muted">—</div></div>', unsafe_allow_html=True)
    else:
        val_loss_col = find_metric_col(log_df, ["val", "valid", "validation", "loss"])
        iou_col = find_metric_col(log_df, ["iou", "jaccard"])
        dice_col = find_metric_col(log_df, ["dice"])
        roc_col = find_metric_col(log_df, ["roc"])
        pr_col = find_metric_col(log_df, ["pr", "ap", "average_precision"])

        best_lines = []
        if val_loss_col is not None:
            best_lines.append(("Валидационный лосс", float(log_df[val_loss_col].min())))
        if iou_col is not None:
            best_lines.append(("IoU", float(log_df[iou_col].max())))
        if dice_col is not None:
            best_lines.append(("Dice", float(log_df[dice_col].max())))
        if roc_col is not None:
            best_lines.append(("ROC-AUC", float(log_df[roc_col].max())))
        if pr_col is not None:
            best_lines.append(("PR-AUC", float(log_df[pr_col].max())))

        if best_lines:
            blocks = []
            for label, value in best_lines[:5]:
                blocks.append(f'<div class="metric-line"><div class="muted">{label}</div><div class="metric-value">{value:.4f}</div></div>')
            st.markdown(
                f'<div class="opaque-card metrics-card"><h3>Лучшие метрики</h3>{"".join(blocks)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="opaque-card metrics-card"><h3>Лучшие метрики</h3><div class="muted">Нет доступных метрик</div></div>',
                unsafe_allow_html=True,
            )

# -----------------------------
# Низ: одна подложка (данные об обучении + параметры в строку)
# -----------------------------
st.divider()

train_df = log_df
epochs_text = "—"
if train_df is not None:
    if epoch_col is not None and pd.api.types.is_numeric_dtype(train_df[epoch_col]):
        epochs_text = str(int(train_df[epoch_col].max()) + 1)
    else:
        epochs_text = str(len(train_df))

def _safe_last(df: pd.DataFrame, col: str) -> str:
    try:
        return str(df.tail(1).iloc[0][col])
    except Exception:
        return "—"

batch_text = "—"
imgsz_text = "—"
lr_text = "—"
opt_text = "—"

if train_df is not None and len(train_df) > 0:
    batch_col = find_metric_col(train_df, ["batch"])
    imgsz_col = find_metric_col(train_df, ["img", "imgsz", "image", "size"])
    lr_col = find_metric_col(train_df, ["lr"])
    opt_col = find_metric_col(train_df, ["optim", "optimizer"])

    if batch_col:
        batch_text = _safe_last(train_df, batch_col)
    if imgsz_col:
        imgsz_text = _safe_last(train_df, imgsz_col)
    if lr_col:
        lr_text = _safe_last(train_df, lr_col)
    if opt_col:
        opt_text = _safe_last(train_df, opt_col)

try:
    _m, model_name_bottom = load_model_cached(MODEL_PY.as_posix(), WEIGHTS_PTH.as_posix())
except Exception:
    model_name_bottom = "—"

kpis_html = (
    '<h3>Данные об обучении и параметры</h3>'
    '<div class="kpi-row">'
    f'<div class="kpi-cell"><div class="kpi-title">Модель</div><div class="kpi-val">{model_name_bottom}</div></div>'
    f'<div class="kpi-cell"><div class="kpi-title">Эпохи</div><div class="kpi-val">{epochs_text}</div></div>'
    f'<div class="kpi-cell"><div class="kpi-title">Размер изображения</div><div class="kpi-val">{imgsz_text}</div></div>'
    f'<div class="kpi-cell"><div class="kpi-title">Batch</div><div class="kpi-val">{batch_text}</div></div>'
    f'<div class="kpi-cell"><div class="kpi-title">Learning rate</div><div class="kpi-val">{lr_text}</div></div>'
    f'<div class="kpi-cell"><div class="kpi-title">Оптимизатор</div><div class="kpi-val">{opt_text}</div></div>'
    '</div>'
)
card_html(kpis_html)

st.divider()
st.markdown(
    '<div class="opaque-card"><p>Работу выполнили студенты Эльбруса — Игорь Никоновский и Сергей Белькин</p></div>',
    unsafe_allow_html=True,
)
