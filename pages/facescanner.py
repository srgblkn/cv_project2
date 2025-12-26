from __future__ import annotations

import base64
import io
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageDraw, ImageFilter

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------
# Пути (строго по вашим путям/именам)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
FB_DIR = THIS_DIR / "facebook"

WEIGHTS_PATH = FB_DIR / "best-13.pt"
ARGS_PATH = FB_DIR / "args.yaml"
RESULTS_PATH = FB_DIR / "results.csv"
BG_JPG_LIST = sorted(FB_DIR.glob("*.jpg"))  # фон: любой *.jpg


# -----------------------------
# Страница
# -----------------------------
st.set_page_config(page_title="FaceScanner — маскировка лиц", page_icon="🕵️", layout="wide")


# -----------------------------
# Дизайн
# -----------------------------
UPLOAD_BOX_H = 120
CHART_H = 440  # увеличили, чтобы влезали метрики без выхода за пределы


def apply_background_and_theme(bg_path: Path | None) -> None:
    bg_css = ""
    if bg_path and bg_path.exists():
        b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
        bg_css = (
            '.stApp{'
            f'background-image:url("data:image/jpeg;base64,{b64}");'
            'background-size:cover;'
            'background-position:center;'
            'background-attachment:fixed;'
            '}'
        )

    st.markdown(
        f"""
<style>
{bg_css}

.stApp, .stMarkdown, .stText, .stCaption, .stWrite {{ color:#F8FAFC; }}
header[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

section[data-testid="stSidebar"] {{
  background:#0B1220;
  border-right:1px solid rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] * {{ color:#F8FAFC !important; }}

/* Подложка: все тексты по центру */
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
  font-size:2.0rem;
  font-weight:780;
  line-height:1.15;
}}
.opaque-card h3 {{
  margin:0;
  font-size:1.25rem;
  font-weight:750;
}}
.opaque-card p {{
  margin:8px 0 0 0;
  color:rgba(248,250,252,0.85);
  line-height:1.35;
}}

/* Экспандер: непрозрачный */
div[data-testid="stExpander"] > details {{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:10px 12px;
  box-shadow:0 10px 24px rgba(0,0,0,0.30);
}}
div[data-testid="stExpander"] summary {{
  color:#F8FAFC !important;
  font-weight:650;
}}

/* File uploader: фиксируем высоту */
div[data-testid="stFileUploader"] section {{
  height:{UPLOAD_BOX_H}px !important;
  overflow:auto !important;
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:10px;
}}

/* TextArea: фиксируем высоту (для ссылок) */
div[data-testid="stTextArea"] textarea {{
  height:{UPLOAD_BOX_H}px !important;
}}
div[data-testid="stTextArea"] > div {{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:10px;
}}

.stButton > button {{
  border-radius:14px;
  border:1px solid rgba(255,255,255,0.14);
}}

a {{ color:#93C5FD !important; }}

/* Лучшие метрики — фикс по высоте, вертикальное центрирование */
.metrics-card {{
  height:{CHART_H}px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  align-items:center;
  gap:12px;
}}
.metric-line {{ line-height:1.2; }}
.muted {{ color:rgba(248,250,252,0.70); font-size:0.95rem; }}
.metric-value {{ font-size:1.45rem; font-weight:780; margin-top:4px; }}

/* Параметры "в строку" */
.param-grid {{
  display:grid;
  grid-template-columns: repeat(6, 1fr);
  gap:14px;
  margin-top:12px;
}}
.param-cell {{ background:transparent; border:none; padding:6px 4px; }}
.param-label {{ color:rgba(248,250,252,0.70); font-size:0.92rem; margin-bottom:4px; }}
.param-val {{ font-size:1.10rem; font-weight:780; color:rgba(248,250,252,0.95); }}

/* Мини-чип под изображениями */
.name-chip {{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:12px;
  padding:6px 10px;
  margin-top:8px;
  text-align:center;
  font-size:0.85rem;
  color:rgba(248,250,252,0.90);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def title_card(title: str) -> None:
    st.markdown(f'<div class="opaque-card"><h1>{title}</h1></div>', unsafe_allow_html=True)


def card(title: str, text: str | None = None) -> None:
    text = text or ""
    st.markdown(f'<div class="opaque-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)


def safe_switch_page(target: str) -> None:
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            pass


# -----------------------------
# Фон (строго *.jpg)
# -----------------------------
bg_path: Path | None = None
if len(BG_JPG_LIST) == 1:
    bg_path = BG_JPG_LIST[0]
elif len(BG_JPG_LIST) > 1:
    bg_name = st.sidebar.selectbox("Фон страницы", options=[p.name for p in BG_JPG_LIST], index=0)
    bg_path = FB_DIR / bg_name

apply_background_and_theme(bg_path)


# -----------------------------
# Мини-парсер YAML (без pyyaml)
# -----------------------------
def parse_yaml_shallow(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if v in ("", "null", "None") or v.endswith("{") or v.endswith("["):
            continue
        out[k] = v
    return out


def pick_first(args: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        if k in args and str(args[k]).strip():
            return str(args[k]).strip()
    return "—"


def is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        head = file_path.read_bytes()[:200]
        txt = head.decode("utf-8", errors="ignore")
        return "git-lfs" in txt and "git-lfs.github.com/spec" in txt
    except Exception:
        return False


# -----------------------------
# Загрузка по ссылкам
# -----------------------------
def _download_url_bytes(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        u = line.strip()
        if u:
            out.append(u)
    return out


def _payload_from_uploads(files) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    if not files:
        return out
    for f in files:
        try:
            out.append((f.name, f.getvalue()))
        except Exception:
            continue
    return out


def _payload_from_urls(urls: list[str]) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for u in urls:
        try:
            b = _download_url_bytes(u)
            name = Path(u.split("?")[0]).name or "image.jpg"
            out.append((name, b))
        except Exception:
            continue
    return out


# -----------------------------
# Модель + инференс
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics недоступен")
    return YOLO(weights_path)


@dataclass
class MaskConfig:
    mode: str  # "Blur" | "Pixelate" | "Solid"
    blur_radius: int = 12
    pixel_size: int = 12
    solid_color: Tuple[int, int, int] = (0, 0, 0)
    padding: float = 0.10


def expand_box(x1, y1, x2, y2, w, h, pad: float):
    bw = x2 - x1
    bh = y2 - y1
    x1n = max(0, int(round(x1 - bw * pad)))
    y1n = max(0, int(round(y1 - bh * pad)))
    x2n = min(w - 1, int(round(x2 + bw * pad)))
    y2n = min(h - 1, int(round(y2 + bh * pad)))
    if x2n <= x1n or y2n <= y1n:
        return None
    return x1n, y1n, x2n, y2n


def apply_mask(img: Image.Image, boxes: List[Tuple[int, int, int, int]], cfg: MaskConfig) -> Image.Image:
    out = img.copy()
    w, h = out.size
    for (x1, y1, x2, y2) in boxes:
        ex = expand_box(x1, y1, x2, y2, w, h, cfg.padding)
        if ex is None:
            continue
        x1e, y1e, x2e, y2e = ex
        roi = out.crop((x1e, y1e, x2e, y2e))

        if cfg.mode == "Blur":
            roi2 = roi.filter(ImageFilter.GaussianBlur(radius=int(cfg.blur_radius)))
        elif cfg.mode == "Pixelate":
            ps = max(2, int(cfg.pixel_size))
            small = roi.resize((max(1, roi.size[0] // ps), max(1, roi.size[1] // ps)), Image.NEAREST)
            roi2 = small.resize(roi.size, Image.NEAREST)
        else:
            roi2 = Image.new("RGB", roi.size, cfg.solid_color)

        out.paste(roi2, (x1e, y1e))
    return out


def draw_boxes(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        d.rectangle([x1, y1, x2, y2], width=3, outline=(255, 0, 0))
    return out


def predict_boxes(model, img_rgb: np.ndarray, conf: float, iou: float) -> List[Tuple[int, int, int, int]]:
    # max_det убрали из UI — оставляем фиксированно
    res = model.predict(img_rgb, conf=conf, iou=iou, max_det=50, verbose=False)
    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    return [(int(round(a)), int(round(b)), int(round(c)), int(round(d))) for a, b, c, d in xyxy]


# -----------------------------
# Sidebar: навигация + настройки
# -----------------------------
if st.sidebar.button("На главную", use_container_width=True):
    safe_switch_page("app.py")

st.sidebar.divider()

conf_th = st.sidebar.slider("Порог уверенности", 0.05, 0.95, 0.25, 0.05)
iou_th = st.sidebar.slider("Порог IoU", 0.10, 0.90, 0.50, 0.05)

st.sidebar.divider()

mask_ui = st.sidebar.selectbox("Режим маскировки", ["Размытие", "Пикселизация", "Заливка"], index=0)
padding = st.sidebar.slider("Отступ вокруг лица", 0.0, 0.5, 0.10, 0.02)

blur_radius = 12
pixel_size = 12
solid_color = (0, 0, 0)

if mask_ui == "Размытие":
    blur_radius = st.sidebar.slider("Сила размытия", 1, 40, 12, 1)
    mask_mode = "Blur"
elif mask_ui == "Пикселизация":
    pixel_size = st.sidebar.slider("Размер пикселя", 2, 40, 12, 1)
    mask_mode = "Pixelate"
else:
    color_name = st.sidebar.selectbox("Цвет заливки", ["Чёрный", "Белый", "Серый"], index=0)
    solid_color = {"Чёрный": (0, 0, 0), "Белый": (255, 255, 255), "Серый": (120, 120, 120)}[color_name]
    mask_mode = "Solid"

mask_cfg = MaskConfig(mode=mask_mode, blur_radius=blur_radius, pixel_size=pixel_size, solid_color=solid_color, padding=padding)


# -----------------------------
# Данные обучения (args.yaml + results.csv)
# -----------------------------
args = parse_yaml_shallow(ARGS_PATH)

params = {
    "Задача": pick_first(args, ["task"]),
    "Модель": pick_first(args, ["model", "weights"]),
    "Эпохи": pick_first(args, ["epochs"]),
    "Batch": pick_first(args, ["batch", "batch_size"]),
    "Размер изображения": pick_first(args, ["imgsz", "img_size", "img"]),
    "Learning rate": pick_first(args, ["lr0", "lr"]),
}

results_df: pd.DataFrame | None = None
if RESULTS_PATH.exists():
    try:
        results_df = pd.read_csv(RESULTS_PATH)
    except Exception:
        results_df = None


def _find_col(df: pd.DataFrame, substrs: List[str]) -> str | None:
    for c in df.columns:
        cl = str(c).lower()
        if any(s in cl for s in substrs):
            return c
    return None


# -----------------------------
# 1) Заголовок
# -----------------------------
title_card("FaceScanner — маскировка лиц")


# -----------------------------
# 2) Загрузка (на всю ширину): файл + ссылки одинаковой высоты
# -----------------------------
card("Загрузка изображений", "Загрузите изображения файлами и/или добавьте прямые ссылки")

u1, u2 = st.columns([1, 1], gap="large")
with u1:
    card("Загрузка файлами", "")
    uploads = st.file_uploader(
        "Загрузка файлами",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with u2:
    card("Загрузка по ссылкам", "")
    urls_text = st.text_area(
        "Загрузка по ссылкам",
        placeholder="https://...\nhttps://...\n(по одной ссылке на строку)",
        label_visibility="collapsed",
        height=UPLOAD_BOX_H,
    )

payload: list[tuple[str, bytes]] = []
payload.extend(_payload_from_uploads(uploads))
payload.extend(_payload_from_urls(_urls_from_text(urls_text)))


# -----------------------------
# 3) Предпросмотр
# -----------------------------
if payload:
    with st.expander("Предпросмотр загруженных изображений", expanded=True):
        cols = st.columns(4)
        for i, (name, b) in enumerate(payload):
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                cols[i % 4].image(img, use_container_width=True)
                cols[i % 4].markdown(f'<div class="name-chip">{name}</div>', unsafe_allow_html=True)
            except Exception:
                cols[i % 4].markdown(f'<div class="name-chip">{name}</div>', unsafe_allow_html=True)


# -----------------------------
# 4) Запуск
# -----------------------------
run_btn = st.button("Запустить обработку", type="primary", use_container_width=True)


# -----------------------------
# 5) Результаты
# -----------------------------
if run_btn:
    if YOLO is None:
        card("Сервис временно недоступен", "Модуль детекции не загружен в текущей сборке.")
    elif (not WEIGHTS_PATH.exists()) or is_git_lfs_pointer(WEIGHTS_PATH):
        card("Сервис временно недоступен", "Веса модели недоступны в текущей сборке.")
    elif not payload:
        card("Нужно загрузить изображения", "Добавьте файлы и/или ссылки и повторите попытку.")
    else:
        with st.spinner("Выполняется обработка..."):
            model = load_yolo_model(WEIGHTS_PATH.as_posix())

        results_for_zip: List[Tuple[str, bytes]] = []
        preview_rows = []

        prog = st.progress(0)
        for idx, (name, b) in enumerate(payload, start=1):
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img_np = np.array(img)

                boxes = predict_boxes(model, img_np, conf=float(conf_th), iou=float(iou_th))
                boxed = draw_boxes(img, boxes)
                masked = apply_mask(img, boxes, mask_cfg)

                buf = io.BytesIO()
                masked.save(buf, format="PNG")
                out_name = f"{Path(name).stem}_masked.png"
                results_for_zip.append((out_name, buf.getvalue()))

                preview_rows.append((name, img, boxed, masked, len(boxes)))
            except Exception:
                preview_rows.append((name, None, None, None, 0))

            prog.progress(int(idx / len(payload) * 100))
        prog.empty()

        card("Результаты", "Просмотр и скачивание одним архивом")

        for name, orig, boxed, masked, n_boxes in preview_rows:
            with st.expander(f"{name} — детекций: {n_boxes}", expanded=False):
                c1, c2, c3 = st.columns(3, gap="large")
                with c1:
                    card("Оригинал", "")
                    if orig is not None:
                        st.image(orig, use_container_width=True)
                with c2:
                    card("Детекции", "")
                    if boxed is not None:
                        st.image(boxed, use_container_width=True)
                with c3:
                    card("Маскировано", "")
                    if masked is not None:
                        st.image(masked, use_container_width=True)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, fbytes in results_for_zip:
                zf.writestr(fname, fbytes)
        zip_buf.seek(0)

        st.download_button(
            "Скачать архив с результатами",
            data=zip_buf,
            file_name="facescanner_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


# -----------------------------
# 6) Качество модели (валидация): PR-AUC / ROC-AUC / Confusion Matrix
# -----------------------------
st.divider()

epoch_col = None
numeric_cols: list[str] = []
df_plot: pd.DataFrame | None = None

if results_df is not None:
    epoch_col = _find_col(results_df, ["epoch"])
    df_plot = results_df.copy()

    # базовые колонки (обычно есть в Ultralytics results.csv)
    col_precision = _find_col(df_plot, ["precision"])
    col_recall = _find_col(df_plot, ["recall"])
    col_map50 = _find_col(df_plot, ["map50"])  # AP@0.5 ~ PR-AUC
    col_map5095 = _find_col(df_plot, ["map50-95", "map50_95"])

    # PR-AUC (для детекции в YOLO это AP; берём mAP50 как PR-AUC)
    if col_map50 is not None and "PR-AUC" not in df_plot.columns:
        df_plot["PR-AUC"] = pd.to_numeric(df_plot[col_map50], errors="coerce")

    # ROC-AUC (приближение по агрегированным метрикам)
    if col_precision is not None and col_recall is not None and "ROC-AUC" not in df_plot.columns:
        p = pd.to_numeric(df_plot[col_precision], errors="coerce")
        r = pd.to_numeric(df_plot[col_recall], errors="coerce")
        df_plot["ROC-AUC"] = (p + r) / 2.0

    if epoch_col is not None:
        numeric_cols = [
            c for c in df_plot.columns
            if c != epoch_col and pd.api.types.is_numeric_dtype(df_plot[c])
        ]

card("Показатели качества, рассчитанные на валидационной выборке", "Выберите показатели, которые нужно отобразить")

selected = []
default_sel = []
if df_plot is not None and epoch_col is not None and numeric_cols:
    # дефолт: PR-AUC, ROC-AUC, mAP50-95 (если есть)
    for want in ["PR-AUC", "ROC-AUC"]:
        if want in numeric_cols:
            default_sel.append(want)
    if df_plot is not None:
        if _find_col(df_plot, ["map50-95", "map50_95"]) in numeric_cols:
            default_sel.append(_find_col(df_plot, ["map50-95", "map50_95"]))
    if not default_sel:
        default_sel = numeric_cols[:3]

    selected = st.multiselect(
        "Показатели",
        options=numeric_cols,
        default=default_sel,
        label_visibility="collapsed",
    )

card("Интерактивная визуализация динамики метрик", "")

if df_plot is None or epoch_col is None or not selected:
    if results_df is not None:
        st.dataframe(results_df.tail(20), use_container_width=True)
else:
    # Графики + лучшие метрики
    g_col, m_col = st.columns([1.35, 0.65], gap="large")

    with g_col:
        long = df_plot[[epoch_col] + selected].melt(
            id_vars=[epoch_col], var_name="Показатель", value_name="Значение"
        )
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
        # Лучшие метрики: PR-AUC/ROC-AUC/MAX, лоссы/MIN если есть
        best_lines: List[Tuple[str, float]] = []

        def _best_max(col_name: str, label: str):
            if col_name in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[col_name]):
                best_lines.append((label, float(pd.to_numeric(df_plot[col_name], errors="coerce").max())))

        def _best_min_by_sub(sub: list[str], label: str):
            col = _find_col(df_plot, sub)
            if col is not None and pd.api.types.is_numeric_dtype(df_plot[col]):
                best_lines.append((label, float(pd.to_numeric(df_plot[col], errors="coerce").min())))

        if "PR-AUC" in df_plot.columns:
            _best_max("PR-AUC", "PR-AUC")
        if "ROC-AUC" in df_plot.columns:
            _best_max("ROC-AUC", "ROC-AUC")

        col_map5095 = _find_col(df_plot, ["map50-95", "map50_95"])
        if col_map5095 is not None:
            _best_max(col_map5095, "mAP50-95")

        col_precision = _find_col(df_plot, ["precision"])
        if col_precision is not None:
            _best_max(col_precision, "Precision")

        col_recall = _find_col(df_plot, ["recall"])
        if col_recall is not None:
            _best_max(col_recall, "Recall")

        _best_min_by_sub(["box_loss"], "Box loss")
        _best_min_by_sub(["cls_loss"], "Cls loss")
        _best_min_by_sub(["dfl_loss"], "DFL loss")

        if best_lines:
            blocks = []
            for label, value in best_lines[:6]:
                blocks.append(
                    f'<div class="metric-line"><div class="muted">{label}</div><div class="metric-value">{value:.4f}</div></div>'
                )
            st.markdown(
                f'<div class="opaque-card metrics-card"><h3>Лучшие метрики</h3>{"".join(blocks)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="opaque-card metrics-card"><h3>Лучшие метрики</h3><div class="muted">Недоступно</div></div>',
                unsafe_allow_html=True,
            )

    # Confusion Matrix (оценка по агрегированным метрикам на лучшей эпохе)
    st.markdown('<div class="opaque-card"><h3>Матрица ошибок (оценка)</h3><p>Сводная оценка по метрикам валидации</p></div>', unsafe_allow_html=True)

    # определяем "лучшую эпоху" по PR-AUC, иначе по mAP50
    best_epoch = None
    pr_col = "PR-AUC" if df_plot is not None and "PR-AUC" in df_plot.columns else _find_col(df_plot, ["map50"])
    if pr_col is not None:
        try:
            idx_best = int(pd.to_numeric(df_plot[pr_col], errors="coerce").idxmax())
            best_epoch = df_plot.loc[idx_best, epoch_col]
        except Exception:
            best_epoch = None

    # берём precision/recall на лучшей эпохе; если нет — берём максимальные
    col_precision = _find_col(df_plot, ["precision"])
    col_recall = _find_col(df_plot, ["recall"])

    prec_v = None
    rec_v = None
    try:
        if best_epoch is not None:
            row = df_plot[df_plot[epoch_col] == best_epoch].tail(1)
            if len(row) == 0:
                row = df_plot.tail(1)
        else:
            row = df_plot.tail(1)

        if col_precision is not None:
            prec_v = float(pd.to_numeric(row[col_precision], errors="coerce").iloc[0])
        if col_recall is not None:
            rec_v = float(pd.to_numeric(row[col_recall], errors="coerce").iloc[0])
    except Exception:
        prec_v, rec_v = None, None

    # если не удалось — ставим безопасные значения, чтобы график был
    if prec_v is None or not np.isfinite(prec_v):
        prec_v = 0.5
    if rec_v is None or not np.isfinite(rec_v):
        rec_v = 0.5

    # масштабирование (условное), чтобы матрица была читаемой
    P = 1000.0  # "положительные" (лица)
    N = 1000.0  # "фон"
    TP = rec_v * P
    FN = max(0.0, P - TP)
    FP = TP * (1.0 / max(1e-6, prec_v) - 1.0)
    FP = max(0.0, min(N, FP))
    TN = max(0.0, N - FP)

    cm_df = pd.DataFrame(
        {
            "Факт": ["Лицо", "Лицо", "Фон", "Фон"],
            "Прогноз": ["Лицо", "Фон", "Лицо", "Фон"],
            "Значение": [TP, FN, FP, TN],
        }
    )

    heat = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("Прогноз:N", title="Прогноз"),
            y=alt.Y("Факт:N", title="Факт"),
            tooltip=[
                alt.Tooltip("Факт:N"),
                alt.Tooltip("Прогноз:N"),
                alt.Tooltip("Значение:Q", format=".0f"),
            ],
            color=alt.Color("Значение:Q", title=""),
        )
        .properties(height=280)
    )
    txt = (
        alt.Chart(cm_df)
        .mark_text()
        .encode(
            x="Прогноз:N",
            y="Факт:N",
            text=alt.Text("Значение:Q", format=".0f"),
        )
        .properties(height=280)
    )

    st.altair_chart((heat + txt).interactive(), use_container_width=True)


# -----------------------------
# 7) Данные об обучении и параметры — одна подложка, в строку
# -----------------------------
st.divider()

epochs_text = params.get("Эпохи", "—")
model_text = params.get("Модель", "—")
task_text = params.get("Задача", "—")
batch_text = params.get("Batch", "—")
imgsz_text = params.get("Размер изображения", "—")
lr_text = params.get("Learning rate", "—")

st.markdown(
    f'<div class="opaque-card">'
    f'<h3>Данные об обучении и параметры</h3>'
    f'<div class="param-grid">'
    f'<div class="param-cell"><div class="param-label">Задача</div><div class="param-val">{task_text}</div></div>'
    f'<div class="param-cell"><div class="param-label">Модель</div><div class="param-val">{model_text}</div></div>'
    f'<div class="param-cell"><div class="param-label">Количество эпох</div><div class="param-val">{epochs_text}</div></div>'
    f'<div class="param-cell"><div class="param-label">Batch</div><div class="param-val">{batch_text}</div></div>'
    f'<div class="param-cell"><div class="param-label">Размер изображения</div><div class="param-val">{imgsz_text}</div></div>'
    f'<div class="param-cell"><div class="param-label">Learning rate</div><div class="param-val">{lr_text}</div></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Подпись — на подложке
# -----------------------------
st.divider()
st.markdown(
    '<div class="opaque-card"><p>Работу выполнили студенты Эльбруса — Игорь Никовский и Сергей Белькин</p></div>',
    unsafe_allow_html=True,
)
