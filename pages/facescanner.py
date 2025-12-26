# pages/facescanner.py
from __future__ import annotations

import base64
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------
# Paths (строго по вашим путям/именам)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
FB_DIR = THIS_DIR / "facebook"

WEIGHTS_PATH = FB_DIR / "best-13.pt"
ARGS_PATH = FB_DIR / "args.yaml"
RESULTS_PATH = FB_DIR / "results.csv"
BG_JPG_LIST = sorted(FB_DIR.glob("*.jpg"))  # фон: любой *.jpg


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="FaceScanner — маскировка лиц", page_icon="🕵️", layout="wide")


# -----------------------------
# UI styling: background + opaque cards + high contrast
# -----------------------------
def apply_background_and_contrast(bg_path: Path | None) -> None:
    bg_css = ""
    if bg_path and bg_path.exists():
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

        header[data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        /* Sidebar: непрозрачный */
        section[data-testid="stSidebar"] {{
            background: #0B1220;
            border-right: 1px solid rgba(255,255,255,0.10);
        }}
        section[data-testid="stSidebar"] * {{
            color: #F8FAFC !important;
        }}

        /* Opaque card */
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

        /* Expander: непрозрачный */
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

        /* File uploader: непрозрачный */
        div[data-testid="stFileUploader"] section {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 10px;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.14);
        }}

        a {{
            color: #93C5FD !important;
        }}
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
            st.info("Переход недоступен в этой среде. Используйте меню слева.")
    else:
        st.info("Переход недоступен в текущей версии Streamlit. Используйте меню слева.")


# -----------------------------
# Background picker (строго *.jpg)
# -----------------------------
bg_path: Path | None = None
if len(BG_JPG_LIST) == 1:
    bg_path = BG_JPG_LIST[0]
elif len(BG_JPG_LIST) > 1:
    # Выбор только если есть из чего выбирать
    bg_name = st.sidebar.selectbox("Фон страницы (*.jpg)", options=[p.name for p in BG_JPG_LIST], index=0)
    bg_path = FB_DIR / bg_name

apply_background_and_contrast(bg_path)


# -----------------------------
# Minimal YAML parsing (без pyyaml)
# -----------------------------
def parse_yaml_shallow(path: Path) -> Dict[str, str]:
    """
    Достаём простые key: value из args.yaml (без вложенных структур).
    Этого достаточно для task/model/epochs/batch/imgsz и пары дополнительных.
    """
    out: Dict[str, str] = {}
    if not path.exists():
        return out

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        key, val = s.split(":", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        # пропускаем явно “сложные” блоки
        if val in ("", "null", "None") or val.endswith("{") or val.endswith("["):
            continue
        out[key] = val
    return out


def pick_first(args: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        if k in args and str(args[k]).strip() != "":
            return str(args[k]).strip()
    return "—"


# -----------------------------
# Weights validation (LFS pointer detection)
# -----------------------------
def is_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        head = file_path.read_bytes()[:200]
        txt = head.decode("utf-8", errors="ignore")
        return "git-lfs" in txt and "version https://git-lfs.github.com/spec" in txt
    except Exception:
        return False


def ensure_weights_ok_or_stop(path: Path) -> None:
    if not path.exists():
        st.error(f"Не найден файл весов: `{path.as_posix()}`")
        st.stop()

    if is_git_lfs_pointer(path):
        st.error(
            "Файл `best-13.pt` в деплое похож на Git LFS pointer (это текстовый файл-ссылка, а не веса). "
            "Из-за этого `torch.load()` падает с UnpicklingError. "
            "Нужно, чтобы в репозитории/деплое был именно бинарный файл весов (или хранить веса вне GitHub и скачивать их в рантайме)."
        )
        st.caption(
            "Проверка: откройте `pages/facebook/best-13.pt` в GitHub. "
            "Если там текст с `version https://git-lfs...` — это pointer."
        )
        st.stop()


# -----------------------------
# Model + inference helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics не установлен. Проверь requirements.txt.")
    return YOLO(weights_path)


@dataclass
class MaskConfig:
    mode: str  # Blur / Pixelate / Solid
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


def predict_boxes(model, img_rgb: np.ndarray, conf: float, iou: float, max_det: int) -> List[Tuple[int, int, int, int]]:
    res = model.predict(img_rgb, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    return [(int(round(a)), int(round(b)), int(round(c)), int(round(d))) for a, b, c, d in xyxy]


# -----------------------------
# Header
# -----------------------------
opaque_card(
    "FaceScanner",
    "Детекция лиц и маскировка. Поддерживается пакетная загрузка и скачивание результата одним архивом.",
)

top_l, top_r = st.columns([1, 1], gap="large")
with top_l:
    if st.button("← На главную", use_container_width=True):
        safe_switch_page("app.py")
with top_r:
    if bg_path and bg_path.exists():
        st.download_button(
            "Скачать фон (JPG)",
            data=bg_path.read_bytes(),
            file_name=bg_path.name,
            mime="image/jpeg",
            use_container_width=True,
        )


# -----------------------------
# Sidebar controls (кратко)
# -----------------------------
st.sidebar.markdown("## Настройки")
conf_th = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
iou_th = st.sidebar.slider("IoU", 0.10, 0.90, 0.50, 0.05)
max_det = st.sidebar.number_input("Max detections", min_value=1, max_value=500, value=50, step=1)

st.sidebar.divider()
mask_mode = st.sidebar.selectbox("Маскировка", ["Blur", "Pixelate", "Solid"], index=0)
padding = st.sidebar.slider("Padding", 0.0, 0.5, 0.10, 0.02)

blur_radius = 12
pixel_size = 12
solid_color = (0, 0, 0)

if mask_mode == "Blur":
    blur_radius = st.sidebar.slider("Blur radius", 1, 40, 12, 1)
elif mask_mode == "Pixelate":
    pixel_size = st.sidebar.slider("Pixel size", 2, 40, 12, 1)
else:
    color_name = st.sidebar.selectbox("Solid color", ["Black", "White", "Gray"], index=0)
    solid_color = {"Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (120, 120, 120)}[color_name]

mask_cfg = MaskConfig(mode=mask_mode, blur_radius=blur_radius, pixel_size=pixel_size, solid_color=solid_color, padding=padding)


# -----------------------------
# Right column: Training params + charts (по запросу)
# -----------------------------
args = parse_yaml_shallow(ARGS_PATH)

params_rows = [
    ("Задача", pick_first(args, ["task"])),
    ("Модель", pick_first(args, ["model", "weights"])),
    ("Эпохи", pick_first(args, ["epochs"])),
    ("Batch", pick_first(args, ["batch", "batch_size"])),
    ("Image size", pick_first(args, ["imgsz", "img_size", "img"])),
    ("Learning rate", pick_first(args, ["lr0", "lr"])),
    ("Optimizer", pick_first(args, ["optimizer"])),
]
params_df = pd.DataFrame(params_rows, columns=["Параметр", "Значение"])

results_df: pd.DataFrame | None = None
if RESULTS_PATH.exists():
    try:
        results_df = pd.read_csv(RESULTS_PATH)
    except Exception:
        results_df = None


# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    opaque_card("Загрузка изображений", "Загрузите один или несколько файлов. Ниже появится предпросмотр.")
    uploads = st.file_uploader(
        "Images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Предпросмотр (чтобы “после загрузки фото — видно”)
    if uploads:
        with st.expander("Предпросмотр загруженных файлов", expanded=True):
            cols = st.columns(4)
            for i, up in enumerate(uploads):
                try:
                    img = Image.open(up).convert("RGB")
                    cols[i % 4].image(img, caption=up.name, use_container_width=True)
                except Exception:
                    cols[i % 4].write(up.name)

    run_btn = st.button("Запустить обработку", type="primary", use_container_width=True)

with right:
    opaque_card("Параметры обучения", "Ключевые параметры обучения модели (вытягиваются из args.yaml).")
    st.table(params_df)

    st.divider()
    opaque_card("Графики обучения", "Выберите метрики/лоссы — построим только то, что нужно.")
    if results_df is None:
        st.info("`results.csv` не найден или не читается.")
    else:
        # какие колонки предлагать
        col_epoch = next((c for c in results_df.columns if c.lower() == "epoch"), None)
        if col_epoch is None:
            st.warning("В results.csv нет колонки `epoch`. Показываю таблицу.")
            st.dataframe(results_df.tail(30), use_container_width=True)
        else:
            # кандидаты
            candidates = []
            keys = ["precision", "recall", "map50", "map50-95", "map50_95", "box_loss", "cls_loss", "dfl_loss"]
            for c in results_df.columns:
                cl = c.lower()
                if any(k in cl for k in keys) and c != col_epoch:
                    candidates.append(c)

            # если кандидатов нет — просто таблица
            if not candidates:
                st.dataframe(results_df.tail(30), use_container_width=True)
            else:
                selected = st.multiselect(
                    "Что построить",
                    options=candidates,
                    default=[candidates[0]],
                )
                if selected:
                    chart_df = results_df[[col_epoch] + selected].copy()
                    chart_df = chart_df.set_index(col_epoch)
                    st.line_chart(chart_df, use_container_width=True)

            # короткая “лучшая эпоха” по mAP, если есть
            score_col = next((c for c in results_df.columns if "map50-95" in c.lower() or "map50_95" in c.lower()), None)
            if score_col is None:
                score_col = next((c for c in results_df.columns if "map50" in c.lower()), None)
            if score_col:
                try:
                    best_idx = int(results_df[score_col].astype(float).idxmax())
                    best_epoch = int(results_df.loc[best_idx, col_epoch])
                    best_score = float(results_df.loc[best_idx, score_col])
                    st.success(f"Лучшая эпоха по `{score_col}`: epoch={best_epoch}, score={best_score:.4f}")
                except Exception:
                    pass


# -----------------------------
# Inference
# -----------------------------
if run_btn:
    if YOLO is None:
        st.error("Не установлен пакет `ultralytics`.")
        st.stop()

    ensure_weights_ok_or_stop(WEIGHTS_PATH)

    if not uploads:
        st.warning("Загрузите хотя бы один файл.")
        st.stop()

    with st.spinner("Загружаю модель..."):
        model = load_yolo_model(WEIGHTS_PATH.as_posix())

    results_for_zip: List[Tuple[str, bytes]] = []
    preview_rows = []

    prog = st.progress(0)
    for idx, up in enumerate(uploads, start=1):
        try:
            img = Image.open(up).convert("RGB")
            img_np = np.array(img)

            boxes = predict_boxes(model, img_np, conf=float(conf_th), iou=float(iou_th), max_det=int(max_det))
            boxed = draw_boxes(img, boxes)
            masked = apply_mask(img, boxes, mask_cfg)

            buf = io.BytesIO()
            masked.save(buf, format="PNG")
            buf.seek(0)

            out_name = f"{Path(up.name).stem}_masked.png"
            results_for_zip.append((out_name, buf.getvalue()))
            preview_rows.append((up.name, img, boxed, masked, len(boxes)))
        except Exception as e:
            st.error(f"Ошибка обработки {up.name}: {e}")

        prog.progress(int(idx / len(uploads) * 100))
    prog.empty()

    opaque_card("Результаты", "Превью и скачивание ZIP.")
    for name, orig, boxed, masked, n_boxes in preview_rows:
        with st.expander(f"{name} — детекций: {n_boxes}", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Оригинал**")
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown("**Детекции**")
                st.image(boxed, use_container_width=True)
            with c3:
                st.markdown("**Маскировано**")
                st.image(masked, use_container_width=True)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        "Скачать ZIP с результатами",
        data=zip_buf,
        file_name="facescanner_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
