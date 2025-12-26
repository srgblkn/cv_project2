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
            pass


apply_background(BG_JPG)


# -----------------------------
# Загрузка по ссылкам
# -----------------------------
def _download_url_bytes(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for line in text.splitlines():
        u = line.strip()
        if not u:
            continue
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


def _load_masks_from_uploads(files) -> list[tuple[str, Image.Image]]:
    out = []
    if not files:
        return out
    for f in files:
        try:
            f.seek(0)
            img = Image.open(f)
            f.seek(0)
            out.append((f.name, img))
        except Exception:
            continue
    return out


def _load_masks_from_urls(urls: list[str]) -> list[tuple[str, Image.Image]]:
    out = []
    for u in urls:
        try:
            b = _download_url_bytes(u)
            img = Image.open(io.BytesIO(b))
            name = Path(u.split("?")[0]).name or "mask.png"
            out.append((name, img))
        except Exception:
            continue
    return out


def _stem(name: str) -> str:
    return Path(name).stem.lower().strip()


def _align_masks_to_images(
    images: list[tuple[str, Image.Image]],
    masks: list[tuple[str, Image.Image]],
) -> dict[str, Image.Image]:
    """
    Возвращает словарь: image_name -> mask_image
    Стратегия:
      1) если кол-во масок == кол-ву изображений -> по порядку
      2) иначе -> по совпадению stem
    """
    out = {}
    if not images or not masks:
        return out

    if len(images) == len(masks):
        for (iname, _), (_, mimg) in zip(images, masks):
            out[iname] = mimg
        return out

    mask_map = {_stem(n): img for n, img in masks}
    for iname, _ in images:
        s = _stem(iname)
        if s in mask_map:
            out[iname] = mask_map[s]
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

    # иначе берём первый класс nn.Module, объявленный в этом модуле
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

    # типовые параметры
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
        # если вдруг многоклассовая — берём канал с максимальной вероятностью !=0
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


def binarize_mask_img(mask_img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """
    Приводим эталонную маску к бинарной [0,1] и заданному размеру.
    """
    m = mask_img.convert("L").resize(size, resample=Image.NEAREST)
    arr = np.array(m).astype(np.float32)
    return (arr > 127).astype(np.float32)


# -----------------------------
# Метрики (numpy-only): Confusion Matrix + ROC-AUC + PR-AUC
# -----------------------------
def confusion_counts(y_true01: np.ndarray, y_pred01: np.ndarray) -> tuple[int, int, int, int]:
    yt = (y_true01 > 0.5).astype(np.uint8).ravel()
    yp = (y_pred01 > 0.5).astype(np.uint8).ravel()

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tp, fp, fn, tn


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def roc_auc_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC через ранги (Mann–Whitney U), без sklearn.
    """
    y_true = y_true.astype(np.uint8).ravel()
    y_score = y_score.astype(np.float64).ravel()

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # учёт тай-значений (усреднение рангов)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = float(np.mean(ranks[order[i:j + 1]]))
            ranks[order[i:j + 1]] = avg
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def pr_auc_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    PR-AUC (Average Precision) без sklearn.
    """
    y_true = y_true.astype(np.uint8).ravel()
    y_score = y_score.astype(np.float64).ravel()

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # AP = sum_{k} precision(k) * (recall(k) - recall(k-1))
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum(precision * (recall - recall_prev))
    return float(ap)


def sample_pixels(y_true01: np.ndarray, y_score01: np.ndarray, max_points: int = 200_000, seed: int = 42):
    yt = y_true01.ravel()
    ys = y_score01.ravel()
    n = yt.size
    if n <= max_points:
        return yt, ys
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return yt[idx], ys[idx]


# -----------------------------
# Заголовок
# -----------------------------
st.markdown("## Сегментация аэрокосмических снимков")

top_left, top_right = st.columns([1, 1], gap="large")
with top_left:
    opaque_card("Загрузка", "Вы можете загрузить изображения файлами и/или вставить прямые ссылки (несколько строк).")
with top_right:
    if st.button("← На главную", use_container_width=True):
        safe_switch_page("app.py")


# -----------------------------
# Sidebar: настройки
# -----------------------------
st.sidebar.markdown("## Настройки")
preset = st.sidebar.selectbox("Режим", ["Быстро", "Сбалансировано", "Максимум"], index=1)
size_map = {"Быстро": 256, "Сбалансировано": 384, "Максимум": 512}
img_size = int(size_map[preset])

threshold = st.sidebar.slider("Порог маски", 0.05, 0.95, 0.50, 0.05)
alpha = st.sidebar.slider("Прозрачность наложения", 0.10, 0.90, 0.45, 0.05)

export_mode = st.sidebar.selectbox("Экспорт", ["ZIP (маски + наложение)", "ZIP (только маски)"], index=0)


# -----------------------------
# Лог обучения: интерактивные графики
# -----------------------------
log_df = None
try:
    log_df = pd.read_csv(TRAIN_LOG)
except Exception:
    log_df = None

charts_left, charts_right = st.columns([1.05, 0.95], gap="large")

with charts_left:
    opaque_card("Качество обучения", "Интерактивные графики по журналу обучения.")
    if log_df is not None and len(log_df) > 0:
        epoch_col = next((c for c in log_df.columns if c.lower() in ("epoch", "step")), None)
        if epoch_col is None:
            st.dataframe(log_df.tail(30), use_container_width=True)
        else:
            numeric_cols = [
                c for c in log_df.columns
                if c != epoch_col and pd.api.types.is_numeric_dtype(log_df[c])
            ]
            if numeric_cols:
                selected = st.multiselect("Показатели", options=numeric_cols, default=numeric_cols[:2])
                if selected:
                    long = log_df[[epoch_col] + selected].melt(
                        id_vars=[epoch_col], var_name="Показатель", value_name="Значение"
                    )
                    base = (
                        alt.Chart(long)
                        .mark_line()
                        .encode(
                            x=alt.X(f"{epoch_col}:Q", title="Эпоха"),
                            y=alt.Y("Значение:Q", title="Значение"),
                            color=alt.Color("Показатель:N", title=""),
                            tooltip=[alt.Tooltip(f"{epoch_col}:Q", title="Эпоха"),
                                     alt.Tooltip("Показатель:N", title="Показатель"),
                                     alt.Tooltip("Значение:Q", title="Значение", format=".5f")],
                        )
                        .interactive()
                    )
                    st.altair_chart(base, use_container_width=True)
            else:
                st.dataframe(log_df.tail(30), use_container_width=True)

with charts_right:
    opaque_card("Сводка", "Последняя строка журнала обучения.")
    if log_df is not None and len(log_df) > 0:
        st.dataframe(log_df.tail(1), use_container_width=True, hide_index=True)
    else:
        st.info("Журнал обучения недоступен.")


st.divider()


# -----------------------------
# Загрузка изображений и (опционально) эталонных масок
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    opaque_card("Изображения", "Загрузите файлы и/или вставьте прямые ссылки (каждая ссылка с новой строки).")

    img_files = st.file_uploader(
        "Загрузка изображений файлами",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    img_urls_text = st.text_area(
        "Загрузка изображений по ссылкам (по одной ссылке на строку)",
        height=120,
        placeholder="https://...\nhttps://...",
    )

    images = []
    images.extend(_load_images_from_uploads(img_files))
    images.extend(_load_images_from_urls(_urls_from_text(img_urls_text)))

    if images:
        with st.expander("Предпросмотр изображений", expanded=True):
            cols = st.columns(4)
            for i, (name, img) in enumerate(images):
                cols[i % 4].image(img, caption=name, use_container_width=True)

with right:
    opaque_card("Эталонные маски (необязательно)", "Если загрузить маски, будут рассчитаны ROC-AUC, PR-AUC и матрица ошибок.")

    mask_files = st.file_uploader(
        "Загрузка масок файлами",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    mask_urls_text = st.text_area(
        "Загрузка масок по ссылкам (по одной ссылке на строку)",
        height=120,
        placeholder="https://...\nhttps://...",
    )

    masks = []
    masks.extend(_load_masks_from_uploads(mask_files))
    masks.extend(_load_masks_from_urls(_urls_from_text(mask_urls_text)))

    mask_by_image = _align_masks_to_images(images, masks)


run_btn = st.button("Запустить сегментацию", type="primary", use_container_width=True)


# -----------------------------
# Инференс
# -----------------------------
if run_btn and images:
    import torch

    try:
        with st.spinner("Выполняется сегментация..."):
            model, _ = load_model_cached(MODEL_PY.as_posix(), WEIGHTS_PTH.as_posix())
    except Exception:
        st.error("Не удалось запустить сегментацию. Проверьте конфигурацию сервиса.")
        st.stop()

    results_for_zip = []
    preview_rows = []

    # для метрик (если есть эталоны)
    agg_tp = agg_fp = agg_fn = agg_tn = 0
    all_scores = []
    all_truth = []

    prog = st.progress(0)
    for idx, (name, orig) in enumerate(images, start=1):
        # подготовка
        resized = orig.resize((img_size, img_size), resample=Image.BILINEAR)
        x = to_tensor_rgb(resized)

        with torch.no_grad():
            y = model(x)
        prob = logits_to_prob_binary(y).detach().cpu().numpy()  # HxW float [0..1] (обычно)

        pred_mask01 = (prob > float(threshold)).astype(np.float32)

        # маску возвращаем к исходному размеру для наложения/экспорта
        pred_mask_img = Image.fromarray((pred_mask01 * 255).astype(np.uint8), mode="L").resize(
            orig.size, resample=Image.NEAREST
        )
        pred_mask01_orig = (np.array(pred_mask_img).astype(np.float32) / 255.0)

        overlay = overlay_mask(orig, pred_mask01_orig, alpha=float(alpha))

        stem = Path(name).stem
        if export_mode == "ZIP (маски + наложение)":
            results_for_zip.append((f"{stem}_маска.png", mask_png_bytes(pred_mask01_orig)))
            results_for_zip.append((f"{stem}_наложение.png", img_png_bytes(overlay)))
        else:
            results_for_zip.append((f"{stem}_маска.png", mask_png_bytes(pred_mask01_orig)))

        coverage = float(pred_mask01_orig.mean())
        preview_rows.append((name, orig, overlay, coverage))

        # метрики, если есть эталон
        if name in mask_by_image:
            gt_mask01 = binarize_mask_img(mask_by_image[name], size=orig.size)

            tp, fp, fn, tn = confusion_counts(gt_mask01, pred_mask01_orig)
            agg_tp += tp
            agg_fp += fp
            agg_fn += fn
            agg_tn += tn

            # для AUC берём вероятности (до порога), но на размере orig
            prob_img = Image.fromarray((np.clip(pred_mask01_orig, 0, 1) * 255).astype(np.uint8), mode="L")
            # осторожно: pred_mask01_orig — это уже бинарь; лучше использовать "prob" на исходном размере
            # поэтому пересчитаем prob на orig.size через resize:
            prob_resized_to_orig = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8), mode="L").resize(
                orig.size, resample=Image.BILINEAR
            )
            prob01_orig = (np.array(prob_resized_to_orig).astype(np.float32) / 255.0)

            yt, ys = sample_pixels(gt_mask01, prob01_orig, max_points=200_000, seed=idx + 7)
            all_truth.append(yt)
            all_scores.append(ys)

        prog.progress(int(idx / len(images) * 100))
    prog.empty()

    opaque_card("Результаты", "Ниже — превью и скачивание результата одним архивом.")

    for name, orig, overlay, cov in preview_rows:
        with st.expander(f"{name} — доля маски: {cov * 100:.1f}%", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Оригинал**")
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown("**Наложение маски**")
                st.image(overlay, use_container_width=True)

    # Метрики и confusion matrix только если реально есть эталонные маски
    if mask_by_image:
        opaque_card("Оценка качества", "Метрики рассчитаны по загруженным эталонным маскам (пиксельный уровень).")

        precision = safe_div(agg_tp, (agg_tp + agg_fp))
        recall = safe_div(agg_tp, (agg_tp + agg_fn))
        f1 = safe_div(2 * precision * recall, (precision + recall))
        iou = safe_div(agg_tp, (agg_tp + agg_fp + agg_fn))
        dice = safe_div(2 * agg_tp, (2 * agg_tp + agg_fp + agg_fn))
        accuracy = safe_div((agg_tp + agg_tn), (agg_tp + agg_fp + agg_fn + agg_tn))

        # AUC/PR-AUC по объединённой выборке (сэмпл пикселей)
        roc_auc = 0.0
        pr_auc = 0.0
        if all_truth and all_scores:
            yt = np.concatenate(all_truth)
            ys = np.concatenate(all_scores)
            roc_auc = roc_auc_score_numpy(yt, ys)
            pr_auc = pr_auc_score_numpy(yt, ys)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Точность (Accuracy)", f"{accuracy:.4f}")
            st.metric("ROC-AUC", f"{roc_auc:.4f}")
        with m2:
            st.metric("Precision", f"{precision:.4f}")
            st.metric("PR-AUC", f"{pr_auc:.4f}")
        with m3:
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1", f"{f1:.4f}")

        st.caption(f"IoU: {iou:.4f} • Dice: {dice:.4f}")

        # Confusion Matrix (Altair heatmap)
        cm = pd.DataFrame(
            {
                "Истина": ["Покрытие", "Покрытие", "Фон", "Фон"],
                "Прогноз": ["Покрытие", "Фон", "Покрытие", "Фон"],
                "Значение": [agg_tp, agg_fn, agg_fp, agg_tn],
            }
        )
        heat = (
            alt.Chart(cm)
            .mark_rect()
            .encode(
                x=alt.X("Прогноз:N", title="Прогноз"),
                y=alt.Y("Истина:N", title="Истина"),
                tooltip=[alt.Tooltip("Истина:N"), alt.Tooltip("Прогноз:N"), alt.Tooltip("Значение:Q")],
            )
        )
        text = (
            alt.Chart(cm)
            .mark_text(fontSize=18)
            .encode(
                x="Прогноз:N",
                y="Истина:N",
                text=alt.Text("Значение:Q"),
            )
        )
        st.altair_chart((heat + text).properties(height=240), use_container_width=True)

    # ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        "Скачать ZIP",
        data=zip_buf,
        file_name="результаты_сегментации.zip",
        mime="application/zip",
        use_container_width=True,
    )

elif run_btn and not images:
    st.warning("Добавьте изображения файлами и/или ссылками.")


st.divider()
st.caption("Работу выполнили студенты Эльбруса — Игорь Никоновский и Сергей Белькин")
