import base64
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------
# Paths (строго по вашим директориям/именам)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "cancerbook"

WEIGHTS_PATH = ART_DIR / "best.pt"
ARGS_PATH = ART_DIR / "args.yaml"
RESULTS_PATH = ART_DIR / "results.csv"
BG_JPG_LIST = sorted(ART_DIR.glob("*.jpg"))  # у вас: screen.jpg


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Medical Scan Analyzer", page_icon="🧠", layout="wide")


# -----------------------------
# UI helpers
# -----------------------------
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


def parse_yaml_shallow(path: Path) -> dict:
    """
    Очень простой парсер key: value (без вложенности). Нам достаточно для таблицы ключевых параметров.
    """
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if (not s) or s.startswith("#") or (":" not in s):
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if not v:
            continue
        out[k] = v
    return out


def pick_first(args: dict, keys: list) -> str:
    for k in keys:
        if k in args and str(args[k]).strip():
            return str(args[k]).strip()
    return "—"


def looks_like_lfs_pointer(p: Path) -> bool:
    if not p.exists():
        return False
    head = p.read_bytes()[:200]
    txt = head.decode("utf-8", errors="ignore")
    return ("git-lfs" in txt) and ("git-lfs.github.com/spec" in txt)


def ensure_weights_ok_or_stop(p: Path) -> None:
    if not p.exists():
        st.error(f"Не найден файл весов: `{p.as_posix()}`")
        st.stop()
    if looks_like_lfs_pointer(p):
        st.error("Файл весов выглядит как Git LFS pointer (ссылка), а не бинарный .pt.")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics не установлен.")
    return YOLO(weights_path)


def draw_boxes(img: Image.Image, boxes_xyxy: list, labels: list | None = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        d.rectangle([x1, y1, x2, y2], width=3, outline=(0, 255, 255))
        if labels and i < len(labels):
            d.text((x1 + 4, max(0, y1 - 14)), labels[i], fill=(0, 255, 255))
    return out


def extract_predictions(result):
    """
    Универсально:
    - если есть боксы -> вернём их + подписи
    - если есть probs -> вернём top-5 в таблицу
    """
    boxes_xyxy = []
    box_labels = []
    cls_df = None

    # boxes
    if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy() if getattr(result.boxes, "conf", None) is not None else None
        cls = result.boxes.cls.detach().cpu().numpy() if getattr(result.boxes, "cls", None) is not None else None
        names = getattr(result, "names", None) or {}

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            boxes_xyxy.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
            c = float(conf[i]) if conf is not None else None
            k = int(cls[i]) if cls is not None else None
            name = names.get(k, str(k)) if k is not None else ""
            label = f"{name} {c:.2f}" if (name and c is not None) else (f"{c:.2f}" if c is not None else name)
            box_labels.append(label)

    # probs (classification)
    probs = getattr(result, "probs", None)
    if probs is not None:
        try:
            p = probs.data.detach().cpu().numpy()
            names = getattr(result, "names", None) or {}
            pairs = [(names.get(i, str(i)), float(p[i])) for i in range(len(p))]
            pairs.sort(key=lambda x: x[1], reverse=True)
            cls_df = pd.DataFrame(pairs[:5], columns=["Класс", "Вероятность"])
        except Exception:
            cls_df = None

    return boxes_xyxy, box_labels, cls_df


# -----------------------------
# Background selection
# -----------------------------
bg_path = None
if len(BG_JPG_LIST) == 1:
    bg_path = BG_JPG_LIST[0]
elif len(BG_JPG_LIST) > 1:
    name = st.sidebar.selectbox("Фон страницы (*.jpg)", [p.name for p in BG_JPG_LIST], index=0)
    bg_path = ART_DIR / name

apply_background_and_contrast(bg_path)


# -----------------------------
# Header
# -----------------------------
opaque_card("Medical Scan Analyzer", "Пакетная обработка изображений: локализация зон интереса и оценка класса результата.")

hl, hr = st.columns([1, 1], gap="large")
with hl:
    if st.button("← На главную", use_container_width=True):
        safe_switch_page("app.py")
with hr:
    if bg_path and bg_path.exists():
        st.download_button(
            "Скачать фон (JPG)",
            data=bg_path.read_bytes(),
            file_name=bg_path.name,
            mime="image/jpeg",
            use_container_width=True,
        )


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## Настройки")
conf_th = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
iou_th = st.sidebar.slider("IoU", 0.10, 0.90, 0.50, 0.05)
max_det = st.sidebar.number_input("Max detections", min_value=1, max_value=500, value=50, step=1)
st.sidebar.divider()
show_boxes = st.sidebar.toggle("Показывать боксы (если есть)", value=True)
export_mode = st.sidebar.selectbox("Экспорт", ["ZIP (изображения)", "ZIP (изображения + CSV)"], index=1)


# -----------------------------
# Training params + charts
# -----------------------------
args = parse_yaml_shallow(ARGS_PATH)
params_df = pd.DataFrame(
    [
        ("Задача", pick_first(args, ["task"])),
        ("Модель", pick_first(args, ["model", "weights"])),
        ("Эпохи", pick_first(args, ["epochs"])),
        ("Batch", pick_first(args, ["batch", "batch_size"])),
        ("Image size", pick_first(args, ["imgsz", "img_size", "img"])),
        ("Learning rate", pick_first(args, ["lr0", "lr"])),
        ("Optimizer", pick_first(args, ["optimizer"])),
    ],
    columns=["Параметр", "Значение"],
)

results_df = None
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

    run_btn = st.button("Запустить анализ", type="primary", use_container_width=True)

with right:
    opaque_card("Параметры обучения", "Ключевые параметры обучения модели.")
    st.table(params_df)

    st.divider()
    opaque_card("Графики обучения", "Выберите метрики/лоссы — построим только выбранное.")
    if results_df is None:
        st.info("`results.csv` не найден или не читается.")
    else:
        epoch_col = next((c for c in results_df.columns if c.lower() == "epoch"), None)
        if epoch_col is None:
            st.dataframe(results_df.tail(30), use_container_width=True)
        else:
            keys = ["precision", "recall", "map50", "map50-95", "map50_95", "box_loss", "cls_loss", "dfl_loss"]
            candidates = [c for c in results_df.columns if c != epoch_col and any(k in c.lower() for k in keys)]
            if not candidates:
                st.dataframe(results_df.tail(30), use_container_width=True)
            else:
                selected = st.multiselect("Показатели", options=candidates, default=[candidates[0]])
                if selected:
                    chart = results_df[[epoch_col] + selected].copy().set_index(epoch_col)
                    st.line_chart(chart, use_container_width=True)


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

    try:
        with st.spinner("Загружаю модель..."):
            model = load_yolo_model(WEIGHTS_PATH.as_posix())
    except Exception as e:
        st.error(
            "Не удалось загрузить веса модели.\n\n"
            f"Файл: `{WEIGHTS_PATH.as_posix()}`\n"
            f"Размер: {WEIGHTS_PATH.stat().st_size} bytes\n\n"
            f"Ошибка: {e}"
        )
        st.stop()

    processed = []
    csv_rows = []

    prog = st.progress(0)
    for idx, up in enumerate(uploads, start=1):
        try:
            up.seek(0)
            img = Image.open(up).convert("RGB")
            up.seek(0)

            img_np = np.array(img)
            res = model.predict(img_np, conf=float(conf_th), iou=float(iou_th), max_det=int(max_det), verbose=False)
            r0 = res[0]

            boxes, box_labels, cls_df = extract_predictions(r0)

            view = img
            if show_boxes and boxes:
                view = draw_boxes(img, boxes, box_labels)

            top_class = None
            top_prob = None
            if cls_df is not None and len(cls_df) > 0:
                top_class = str(cls_df.iloc[0]["Класс"])
                top_prob = float(cls_df.iloc[0]["Вероятность"])

            csv_rows.append({"file": up.name, "num_boxes": len(boxes), "top_class": top_class, "top_prob": top_prob})

            buf = io.BytesIO()
            view.save(buf, format="PNG")
            buf.seek(0)

            out_name = f"{Path(up.name).stem}_result.png"
            processed.append((out_name, buf.getvalue()))

            with st.expander(f"{up.name} — результат", expanded=False):
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Исходное**")
                    st.image(img, use_container_width=True)
                with c2:
                    st.markdown("**Результат**")
                    st.image(view, use_container_width=True)
                if cls_df is not None:
                    st.markdown("**Оценка по классам (Top-5)**")
                    st.dataframe(cls_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Ошибка обработки {up.name}: {e}")

        prog.progress(int(idx / len(uploads) * 100))
    prog.empty()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in processed:
            zf.writestr(fname, fbytes)
        if export_mode == "ZIP (изображения + CSV)" and csv_rows:
            zf.writestr("summary.csv", pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8"))

    zip_buf.seek(0)

    st.download_button(
        "Скачать ZIP с результатами",
        data=zip_buf,
        file_name="cancer_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
