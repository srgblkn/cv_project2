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
# Оформление
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
        .opaque-card h1 {{
            margin: 0;
            font-size: 2.0rem;
            font-weight: 780;
            color: #F8FAFC;
            line-height: 1.15;
        }}
        .opaque-card h3 {{
            margin: 0;
            font-size: 1.25rem;
            font-weight: 750;
            color: #F8FAFC;
        }}
        .opaque-card p {{
            margin: 8px 0 0 0;
            color: rgba(248,250,252,0.85);
            line-height: 1.35;
        }}

        /* Центрирование текста в карточке загрузки */
        .centered-text {{
            text-align: center;
        }}

        /* Выравнивание по высоте блоков "Файлы" и "Ссылки" */
        .upload-grid > div {{
            height: 100%;
        }}
        .upload-box {{
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .upload-box .stTextArea, .upload-box .stFileUploader {{
            flex: 1 1 auto;
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


def title_card(title: str) -> None:
    st.markdown(
        f"""
        <div class="opaque-card">
          <h1>{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def opaque_card(title: str, text: str, centered: bool = False) -> None:
    cls = "centered-text" if centered else ""
    st.markdown(
        f"""
        <div class="opaque-card {cls}">
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
            pass


apply_background(BG_JPG)


# -----------------------------
# Сайдбар: навигация + настройки
# -----------------------------
st.sidebar.markdown("## Навигация")
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
# Загрузка по ссылкам
# -----------------------------
def _download_url_bytes(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for line in text.splitlines():
        u = line.strip()
        if u:
            out.append(u)
    return out


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
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt

        cleaned = {}
        for k, v in state.items():
            if isinstance(k, str):
                cleaned[k.replace("module.", "")] = v

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
# Сегментация: постобработка и визуализация
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
# Качество модели по валидной выборке: лог + артефакты (если есть)
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


def find_artifact_image(dir_path: Path, stems: list[str]) -> Path | None:
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for stem in stems:
        for ext in exts:
            p = dir_path / f"{stem}{ext}"
            if p.exists():
                return p
    return None


# -----------------------------
# 1) Заголовок
# -----------------------------
title_card("Сегментация аэрокосмических снимков")


# -----------------------------
# 2) Загрузка на всю ширину
# -----------------------------
opaque_card(
    "Загрузка изображений",
    "Загрузите изображения файлами и/или добавьте прямые ссылки",
    centered=True,
)

st.markdown('<div class="upload-grid">', unsafe_allow_html=True)
u_cols = st.columns([1, 1], gap="large")
with u_cols[0]:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    img_files = st.file_uploader(
        "Загрузка файлами",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with u_cols[1]:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    img_urls_text = st.text_area(
        "Загрузка по ссылкам",
        height=165,  # подгоняем высоту под визуальное равенство
        placeholder="https://...\nhttps://...",
    )
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

images = []
images.extend(_load_images_from_uploads(img_files))
images.extend(_load_images_from_urls(_urls_from_text(img_urls_text)))


# -----------------------------
# 3) Предпросмотр
# -----------------------------
if images:
    with st.expander("Предпросмотр загруженных изображений", expanded=True):
        cols = st.columns(4)
        for i, (name, img) in enumerate(images):
            cols[i % 4].image(img, caption=name, use_container_width=True)


# -----------------------------
# 4) Кнопка запуска
# -----------------------------
run_btn = st.button("Запустить сегментацию", type="primary", use_container_width=True)


# -----------------------------
# 5) Результаты
# -----------------------------
if run_btn:
    if not images:
        st.warning("Добавьте изображения файлами и/или ссылками.")
    else:
        import torch

        with st.spinner("Выполняется сегментация..."):
            model, model_name = load_model_cached(MODEL_PY.as_posix(), WEIGHTS_PTH.as_posix())

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

        opaque_card("Результаты", "Превью и выгрузка результатов одним архивом.")
        for name, orig, overlay, cov in preview_rows:
            with st.expander(f"{name} — доля маски: {cov * 100:.1f}%", expanded=False):
                c1, c2 = st.columns(2, gap="large")
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
            "Скачать архив с результатами",
            data=zip_buf,
            file_name="результаты_сегментации.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # Параметры модели — сразу после результата (логически ожидаемо пользователю)
        st.divider()
        opaque_card("Параметры модели", "Ключевые параметры архитектуры и инференса.")

        def count_params(m) -> int:
            try:
                return int(sum(p.numel() for p in m.parameters()))
            except Exception:
                return 0

        # пытаемся вытащить in/out каналы из сигнатуры класса
        # (без гарантий — поэтому выводим только если удастся)
        in_ch = None
        out_ch = None
        try:
            sig = inspect.signature(type(model).__init__)
            for p in list(sig.parameters.values())[1:]:
                if p.name in ("in_channels", "n_channels", "channels"):
                    in_ch = 3
                if p.name in ("out_channels", "n_class", "n_classes", "num_classes", "classes"):
                    out_ch = 1
        except Exception:
            pass

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Архитектура", str(model_name))
        p2.metric("Размер инференса", f"{img_size}×{img_size}")
        p3.metric("Порог маски", f"{float(threshold):.2f}")
        p4.metric("Параметров (шт.)", f"{count_params(model):,}".replace(",", " "))

        if in_ch is not None or out_ch is not None:
            s = []
            if in_ch is not None:
                s.append(f"Входных каналов: {in_ch}")
            if out_ch is not None:
                s.append(f"Выходных каналов/классов: {out_ch}")
            st.caption(" • ".join(s))


# -----------------------------
# 6) Модель и качество (ниже всего)
# -----------------------------
st.divider()
opaque_card("Показатели качества, рассчитаные на валидационной выборке", "Графики и ключевые метрики обучения.")

log_df = load_training_log(TRAIN_LOG)
if log_df is None:
    st.info("Данные обучения недоступны.")
else:
    epoch_col = pick_epoch_col(log_df)

    val_loss_col = find_metric_col(log_df, ["val", "valid", "validation", "loss"])
    iou_col = find_metric_col(log_df, ["iou", "jaccard"])

    # Блок: графики (слева) + лучшие метрики (справа)
    g_col, m_col = st.columns([1.35, 0.65], gap="large")

    with g_col:
        st.markdown("### Графики обучения")
        if epoch_col is None:
            st.dataframe(log_df.tail(30), use_container_width=True)
        else:
            numeric_cols = [
                c for c in log_df.columns
                if c != epoch_col and pd.api.types.is_numeric_dtype(log_df[c])
            ]
            if numeric_cols:
                selected = st.multiselect("Показатели для графика", options=numeric_cols, default=numeric_cols[:3])
                if selected:
                    long = log_df[[epoch_col] + selected].melt(
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
                        .properties(height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(log_df.tail(30), use_container_width=True)

    with m_col:
        st.markdown("### Лучшие метрики")
        # Лучшими считаем: минимум val_loss и максимум IoU (если есть)
        if val_loss_col is not None:
            best_val_loss = float(log_df[val_loss_col].min())
            st.metric("Валидационный лосс", f"{best_val_loss:.4f}")
        if iou_col is not None:
            best_iou = float(log_df[iou_col].max())
            st.metric("IoU", f"{best_iou:.4f}")

        # Если есть дополнительные метрики — покажем компактно
        dice_col = find_metric_col(log_df, ["dice"])
        if dice_col is not None:
            best_dice = float(log_df[dice_col].max())
            st.metric("Dice", f"{best_dice:.4f}")

        roc_col = find_metric_col(log_df, ["roc"])
        pr_col = find_metric_col(log_df, ["pr", "ap", "average_precision"])
        if roc_col is not None:
            st.metric("ROC-AUC", f"{float(log_df[roc_col].max()):.4f}")
        if pr_col is not None:
            st.metric("PR-AUC", f"{float(log_df[pr_col].max()):.4f}")

    # Confusion matrix и кривые — как готовые артефакты обучения, если сохранены
    st.markdown("### Диагностика качества")
    cm_img = find_artifact_image(ART_DIR, ["confusion_matrix", "confmat", "cm"])
    if cm_img:
        st.image(Image.open(cm_img), caption="Матрица ошибок (валидационная выборка)", use_container_width=True)

st.divider()
st.caption("Работу выполнили студенты Эльбруса — Игорь Никоновский и Сергей Белькин")
