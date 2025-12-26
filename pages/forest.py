import base64
import io
import zipfile
import inspect
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# -----------------------------
# Paths (строго по вашей структуре)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "forrest"

BG_JPG = ART_DIR / "forrest.jpg"
MODEL_PY = ART_DIR / "model_class.py"
WEIGHTS_PTH = ART_DIR / "model_unet.pth"
TRAIN_LOG = ART_DIR / "unet_training_log.csv"


# -----------------------------
# Page config (без эмодзи — чтобы исключить проблемы кодировок)
# -----------------------------
st.set_page_config(page_title="Forest Segmentation", page_icon="*", layout="wide")


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
# Import model class from model_class.py (as-is)
# -----------------------------
def import_model_module(model_py: Path):
    if not model_py.exists():
        raise FileNotFoundError(f"Не найден файл: {model_py.as_posix()}")

    spec = importlib.util.spec_from_file_location("forrest_model_class", model_py.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось создать import spec для model_class.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_model_class(module):
    """
    1) Если есть UNet — используем.
    2) Иначе берем первый класс torch.nn.Module, объявленный в модуле.
    """
    import torch.nn as nn

    if hasattr(module, "UNet"):
        cls = getattr(module, "UNet")
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return cls, "UNet"

    candidates = []
    for name, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            # фильтр: хотим классы из этого модуля, а не импортированные
            if getattr(obj, "__module__", "") == module.__name__:
                candidates.append((name, obj))

    if not candidates:
        raise AttributeError("Не найден ни UNet, ни другой nn.Module класс в model_class.py")

    candidates.sort(key=lambda x: x[0].lower())
    return candidates[0][1], candidates[0][0]


def build_model(model_cls):
    """
    Пытаемся создать модель, подставляя типовые параметры, если они есть в __init__.
    """
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.values())[1:]  # без self
    kwargs = {}

    # типовые имена
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

    # 1) пробуем с kwargs
    try:
        return model_cls(**kwargs)
    except Exception:
        pass

    # 2) пробуем без аргументов
    try:
        return model_cls()
    except Exception as e:
        raise RuntimeError(
            f"Не удалось создать экземпляр модели {model_cls.__name__}. "
            f"Проверьте сигнатуру __init__ в model_class.py. Ошибка: {e}"
        )


def load_weights_into_model(model, weights_path: Path):
    import torch

    if not weights_path.exists():
        raise FileNotFoundError(f"Не найден файл весов: {weights_path.as_posix()}")

    ckpt = torch.load(weights_path.as_posix(), map_location="cpu")

    # Если сохранена целиком модель
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        return ckpt

    # dict-форматы
    if isinstance(ckpt, dict):
        state = None
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            # возможно это уже state_dict
            # отсекаем служебные ключи, если они есть
            # (если окажется не state_dict — load_state_dict упадет, мы перехватим)
            state = ckpt

        cleaned = {}
        for k, v in state.items():
            if not isinstance(k, str):
                continue
            cleaned[k.replace("module.", "")] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        return model

    raise RuntimeError("Неподдерживаемый формат weights-файла (не dict и не nn.Module).")


# -----------------------------
# Image utilities
# -----------------------------
def to_tensor_rgb(img: Image.Image):
    import torch

    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return t


def sigmoid_probs(logits):
    import torch

    return torch.sigmoid(logits)


def overlay_mask(img_rgb: Image.Image, mask01: np.ndarray, alpha: float):
    base = img_rgb.convert("RGBA")
    h, w = mask01.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov = np.array(overlay)

    m = mask01 > 0.5
    ov[m] = np.array([46, 204, 113, int(255 * alpha)], dtype=np.uint8)  # green RGBA
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
# Header
# -----------------------------
opaque_card(
    "Forest Segmentation",
    "Сегментация покрытия на спутниковых снимках: маска + наложение, пакетная обработка и скачивание одним ZIP.",
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
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## Настройки")
preset = st.sidebar.selectbox("Режим", ["Fast", "Balanced", "High"], index=1)
size_map = {"Fast": 256, "Balanced": 384, "High": 512}
img_size = int(size_map[preset])

threshold = st.sidebar.slider("Порог маски", 0.05, 0.95, 0.50, 0.05)
alpha = st.sidebar.slider("Прозрачность наложения", 0.10, 0.90, 0.45, 0.05)
export_mode = st.sidebar.selectbox("Экспорт", ["ZIP (маски + overlay)", "ZIP (только маски)"], index=0)


# -----------------------------
# Training log: charts by request
# -----------------------------
log_df = None
if TRAIN_LOG.exists():
    try:
        log_df = pd.read_csv(TRAIN_LOG)
    except Exception:
        log_df = None

top_l, top_r = st.columns([1.0, 1.0], gap="large")

with top_l:
    opaque_card("Параметры обучения", "Сводка по логам обучения.")
    if log_df is None:
        st.info(f"Лог не найден или не читается: `{TRAIN_LOG.as_posix()}`")
    else:
        st.dataframe(log_df.tail(1), use_container_width=True, hide_index=True)

with top_r:
    opaque_card("Графики обучения", "Выберите показатели — отобразим только выбранное.")
    if log_df is None:
        st.info("Нет данных для построения графиков.")
    else:
        epoch_col = next((c for c in log_df.columns if c.lower() in ("epoch", "step")), None)
        if epoch_col is None:
            st.dataframe(log_df.tail(30), use_container_width=True)
        else:
            numeric_cols = [
                c for c in log_df.columns
                if c != epoch_col and pd.api.types.is_numeric_dtype(log_df[c])
            ]
            if not numeric_cols:
                st.dataframe(log_df.tail(30), use_container_width=True)
            else:
                selected = st.multiselect("Показатели", options=numeric_cols, default=numeric_cols[:1])
                if selected:
                    chart = log_df[[epoch_col] + selected].copy().set_index(epoch_col)
                    st.line_chart(chart, use_container_width=True)

st.divider()


# -----------------------------
# Inference UI
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    opaque_card("Загрузка снимков", "Загрузите один или несколько файлов. В конце можно скачать ZIP.")
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
    opaque_card("Артефакты", "Проверка наличия модели/весов/фона/логов.")
    st.write(f"• model_class.py: `{MODEL_PY.exists()}`")
    st.write(f"• model_unet.pth: `{WEIGHTS_PTH.exists()}`")
    st.write(f"• unet_training_log.csv: `{TRAIN_LOG.exists()}`")
    st.write(f"• forrest.jpg: `{BG_JPG.exists()}`")
    st.caption(f"Путь весов: `{WEIGHTS_PTH.as_posix()}`")


# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_py: str, weights_path: str):
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"PyTorch недоступен: {e}")

    module = import_model_module(Path(model_py))
    model_cls, model_name = pick_model_class(module)
    model = build_model(model_cls)
    model = load_weights_into_model(model, Path(weights_path))
    model.eval()
    return model, model_name


# -----------------------------
# Inference
# -----------------------------
if run_btn:
    if not uploads:
        st.warning("Загрузите хотя бы один файл.")
        st.stop()

    if not MODEL_PY.exists():
        st.error(f"Не найден {MODEL_PY.as_posix()}")
        st.stop()

    if not WEIGHTS_PTH.exists():
        st.error(f"Не найден {WEIGHTS_PTH.as_posix()}")
        st.stop()

    try:
        with st.spinner("Загружаю модель и веса..."):
            model, model_name = load_model(MODEL_PY.as_posix(), WEIGHTS_PTH.as_posix())
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.stop()

    import torch

    results_for_zip = []
    preview_rows = []

    prog = st.progress(0)
    for idx, up in enumerate(uploads, start=1):
        try:
            up.seek(0)
            orig = Image.open(up).convert("RGB")
            up.seek(0)

            # инференс на фиксированном размере для стабильности
            resized = orig.resize((img_size, img_size), resample=Image.BILINEAR)

            x = to_tensor_rgb(resized)
            with torch.no_grad():
                y = model(x)

            # ожидаем [1,1,H,W] или [1,H,W]
            y_np = None
            if isinstance(y, (list, tuple)):
                y = y[0]

            if hasattr(y, "dim") and y.dim() == 4:
                y_np = y[0, 0].detach().cpu().numpy()
            elif hasattr(y, "dim") and y.dim() == 3:
                y_np = y[0].detach().cpu().numpy()
            else:
                raise RuntimeError("Неожиданная форма выхода модели. Ожидали [1,1,H,W] или [1,H,W].")

            probs = sigmoid_probs(torch.from_numpy(y_np)).numpy() if not np.issubdtype(y_np.dtype, np.floating) else 1 / (1 + np.exp(-y_np))
            mask01 = (probs > float(threshold)).astype(np.float32)

            # возвращаем маску к размеру оригинала для overlay
            mask_img = Image.fromarray((mask01 * 255).astype(np.uint8), mode="L")
            mask_img = mask_img.resize(orig.size, resample=Image.NEAREST)
            mask01_orig = (np.array(mask_img).astype(np.float32) / 255.0)

            overlay = overlay_mask(orig, mask01_orig, alpha=float(alpha))

            stem = Path(up.name).stem
            if export_mode == "ZIP (маски + overlay)":
                results_for_zip.append((f"{stem}_mask.png", mask_png_bytes(mask01_orig)))
                results_for_zip.append((f"{stem}_overlay.png", img_png_bytes(overlay)))
            else:
                results_for_zip.append((f"{stem}_mask.png", mask_png_bytes(mask01_orig)))

            coverage = float(mask01_orig.mean())
            preview_rows.append((up.name, orig, overlay, coverage))

        except Exception as e:
            st.error(f"Ошибка обработки {up.name}: {e}")

        prog.progress(int(idx / len(uploads) * 100))
    prog.empty()

    opaque_card("Результаты", f"Модель: {model_name}. Превью и экспорт в ZIP.")
    for name, orig, overlay, cov in preview_rows:
        with st.expander(f"{name} — покрытие: {cov * 100:.1f}%", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Оригинал**")
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown("**Наложение маски**")
                st.image(overlay, use_container_width=True)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        "Скачать ZIP",
        data=zip_buf,
        file_name="forrest_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.divider()
st.caption("Работу выполнили студенты Эльбруса — Игорь Никоновский и Сергей Белькин")
