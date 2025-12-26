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
# cv_project2/pages/forest.py
# cv_project2/pages/forrest/{forrest.jpg, model_class.py, model_unet.pth, unet_training_log.csv}
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
            pass


apply_background(BG_JPG)


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

    # сохранена целиком модель
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        return ckpt

    # dict / state_dict
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
    """
    Поддержка частых вариантов:
    - [1,1,H,W] -> sigmoid
    - [1,2,H,W] -> softmax -> prob класса 1
    - [1,H,W]   -> sigmoid
    """
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


def overlay_mask(img_rgb: Image.Image, mask01: np.ndarray, alpha: float):
    base = img_rgb.convert("RGBA")
    h, w = mask01.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov = np.array(overlay)
    m = mask01 > 0.5
    ov[m] = np.array([46, 204, 113, int(255 * alpha)], dtype=np.uint8)
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
# Визуализация качества модели по валидной выборке (по артефактам обучения)
# -----------------------------
def load_training_log(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def pick_epoch_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).lower() in ("epoch", "step"):
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


def find_artifact_csv(dir_path: Path, stems: list[str]) -> Path | None:
    for stem in stems:
        p = dir_path / f"{stem}.csv"
        if p.exists():
            return p
    return None


def show_curve_from_csv(csv_path: Path, x_name: str, y_name: str, title: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if x_name not in df.columns or y_name not in df.columns:
        return

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(f"{x_name}:Q", title=x_name),
            y=alt.Y(f"{y_name}:Q", title=y_name),
            tooltip=[alt.Tooltip(f"{x_name}:Q"), alt.Tooltip(f"{y_name}:Q")],
        )
        .interactive()
        .properties(title=title, height=260)
    )
    st.altair_chart(chart, use_container_width=True)


# -----------------------------
# 1) Заголовок (с подложкой)
# -----------------------------
title_card("Сегментация аэрокосмических снимков")

# Навигация на главную (оставляем аккуратно, без лишних блоков)
nav_cols = st.columns([1, 1])
with nav_cols[0]:
    if st.button("На главную", use_container_width=True):
        safe_switch_page("app.py")


# -----------------------------
# 2) На всю ширину: загрузка изображений (файлом и по ссылке)
# -----------------------------
opaque_card("Загрузка изображений", "Загрузите изображения файлами и/или добавьте прямые ссылки (по одной на строку).")

u_cols = st.columns([1, 1], gap="large")
with u_cols[0]:
    img_files = st.file_uploader(
        "Загрузка файлами",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
with u_cols[1]:
    img_urls_text = st.text_area(
        "Загрузка по ссылкам",
        height=130,
        placeholder="https://...\nhttps://...",
    )

images = []
images.extend(_load_images_from_uploads(img_files))
images.extend(_load_images_from_urls(_urls_from_text(img_urls_text)))


# -----------------------------
# 3) Предпросмотр (ниже)
# -----------------------------
if images:
    with st.expander("Предпросмотр загруженных изображений", expanded=True):
        cols = st.columns(4)
        for i, (name, img) in enumerate(images):
            cols[i % 4].image(img, caption=name, use_container_width=True)


# -----------------------------
# 4) Кнопка запуска (ниже)
# -----------------------------
run_btn = st.button("Запустить сегментацию", type="primary", use_container_width=True)


# -----------------------------
# 5) Результаты (после запуска) + далее описание модели и качество
# -----------------------------
if run_btn:
    if not images:
        st.warning("Добавьте изображения файлами и/или ссылками.")
    else:
        import torch

        with st.spinner("Выполняется сегментация..."):
            model, _model_name = load_model_cached(MODEL_PY.as_posix(), WEIGHTS_PTH.as_posix())

        results_for_zip = []
        preview_rows = []

        prog = st.progress(0)
        for idx, (name, orig) in enumerate(images, start=1):
            resized = orig.resize((384, 384), resample=Image.BILINEAR)  # фиксируем комфортный размер
            x = to_tensor_rgb(resized)

            with torch.no_grad():
                y = model(x)

            prob_small = logits_to_prob_binary(y).detach().cpu().numpy()
            mask_small = (prob_small > 0.50).astype(np.float32)

            # к размеру оригинала
            mask_img = Image.fromarray((mask_small * 255).astype(np.uint8), mode="L").resize(
                orig.size, resample=Image.NEAREST
            )
            mask01_orig = (np.array(mask_img).astype(np.float32) / 255.0)

            overlay = overlay_mask(orig, mask01_orig, alpha=0.45)

            stem = Path(name).stem
            results_for_zip.append((f"{stem}_маска.png", mask_png_bytes(mask01_orig)))
            results_for_zip.append((f"{stem}_наложение.png", img_png_bytes(overlay)))

            coverage = float(mask01_orig.mean())
            preview_rows.append((name, orig, overlay, coverage))

            prog.progress(int(idx / len(images) * 100))
        prog.empty()

        opaque_card("Результаты", "Ниже — превью и выгрузка результатов одним архивом.")
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

    st.divider()

# ---- Блок описания модели и качества (всегда ниже кнопки/результатов) ----
opaque_card("Модель и качество", "Показатели качества рассчитаны на валидационной выборке в ходе обучения.")

log_df = load_training_log(TRAIN_LOG)
if log_df is None:
    st.info("Данные обучения недоступны.")
else:
    epoch_col = pick_epoch_col(log_df)

    # метрики (пытаемся найти валидационные колонки)
    cols_lower = {c: str(c).lower() for c in log_df.columns}

    def find_col(substrs: list[str]) -> str | None:
        for c, cl in cols_lower.items():
            if any(s in cl for s in substrs):
                return c
        return None

    val_loss_col = find_col(["val", "valid", "validation", "loss"])
    dice_col = find_col(["dice"])
    iou_col = find_col(["iou", "jaccard"])
    roc_col = find_col(["roc"])
    pr_col = find_col(["pr", "ap", "average_precision"])

    last = log_df.tail(1).iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    if val_loss_col:
        m1.metric("Валидационный лосс", f"{float(last[val_loss_col]):.4f}")
    if dice_col:
        m2.metric("Dice", f"{float(last[dice_col]):.4f}")
    if iou_col:
        m3.metric("IoU", f"{float(last[iou_col]):.4f}")
    if roc_col:
        m4.metric("ROC-AUC", f"{float(last[roc_col]):.4f}")
    elif pr_col:
        m4.metric("PR-AUC", f"{float(last[pr_col]):.4f}")

    # интерактивные графики
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

    # Confusion matrix / ROC / PR как артефакты обучения (если есть)
    st.markdown("### Диагностика качества")
    cm_img = find_artifact_image(ART_DIR, ["confusion_matrix", "confmat", "cm"])
    if cm_img:
        st.image(Image.open(cm_img), caption="Матрица ошибок (валидационная выборка)", use_container_width=True)

    roc_csv = find_artifact_csv(ART_DIR, ["roc_curve", "roc"])
    pr_csv = find_artifact_csv(ART_DIR, ["pr_curve", "pr"])
    if roc_csv:
        show_curve_from_csv(roc_csv, x_name="fpr", y_name="tpr", title="ROC-кривая (валидационная выборка)")
    if pr_csv:
        show_curve_from_csv(pr_csv, x_name="recall", y_name="precision", title="PR-кривая (валидационная выборка)")

st.divider()
st.caption("Работу выполнили студенты Эльбруса — Игорь Никоновский и Сергей Белькин")
