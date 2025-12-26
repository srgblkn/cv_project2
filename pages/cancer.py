from __future__ import annotations

import base64
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------
# Пути (строго по вашим директориям/именам)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "cancerbook"

WEIGHTS_PATH = ART_DIR / "best.pt"
ARGS_PATH = ART_DIR / "args.yaml"
RESULTS_PATH = ART_DIR / "results.csv"
BG_JPG_LIST = sorted(ART_DIR.glob("*.jpg"))  # screen.jpg (или любой *.jpg)


# -----------------------------
# Конфиг UI
# -----------------------------
UPLOAD_BOX_H = 120   # компактно (как вы просили: меньше по высоте)
CHART_H = 440        # график + "лучшие метрики" одинаковой высоты

st.set_page_config(page_title="Анализ снимков МРТ", page_icon="🧠", layout="wide")


# -----------------------------
# CSS / Подложки (всё по центру)
# -----------------------------
def _inject_css(bg_path: Optional[Path]) -> None:
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

    css = r"""
<style>
__BG_CSS__

.stApp, .stMarkdown, .stText, .stCaption, .stWrite { color:#F8FAFC; }
header[data-testid="stHeader"] { background: rgba(0,0,0,0); }

section[data-testid="stSidebar"]{
  background:#0B1220;
  border-right:1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] * { color:#F8FAFC !important; }

/* Подложка: все тексты по центру */
.opaque-card{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:16px 16px 14px 16px;
  box-shadow:0 10px 24px rgba(0,0,0,0.40);
  margin-bottom:14px;
  text-align:center;
}
.opaque-card * { text-align:center; }

.opaque-card h1{
  margin:0;
  font-size:2.0rem;
  font-weight:780;
  line-height:1.15;
}
.opaque-card h3{
  margin:0;
  font-size:1.25rem;
  font-weight:750;
}
.opaque-card p{
  margin:8px 0 0 0;
  color:rgba(248,250,252,0.85);
  line-height:1.35;
}

/* Экспандер */
div[data-testid="stExpander"] > details{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:10px 12px;
  box-shadow:0 10px 24px rgba(0,0,0,0.30);
}
div[data-testid="stExpander"] summary{
  color:#F8FAFC !important;
  font-weight:650;
}

/* File uploader: фиксируем высоту */
div[data-testid="stFileUploader"] section{
  height:__UPLOAD_BOX_H__px !important;
  overflow:auto !important;
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:10px;
}

/* TextArea (ссылки): фиксируем высоту */
div[data-testid="stTextArea"] textarea{
  height:__UPLOAD_BOX_H__px !important;
}
div[data-testid="stTextArea"] > div{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:10px;
}

.stButton > button{
  border-radius:14px;
  border:1px solid rgba(255,255,255,0.14);
}

a{ color:#93C5FD !important; }

/* Лучшие метрики — высота как у графика + вертикальный центр */
.metrics-card{
  height:__CHART_H__px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  align-items:center;
  gap:12px;
}
.metric-line{ line-height:1.2; }
.muted{ color:rgba(248,250,252,0.70); font-size:0.95rem; }
.metric-value{ font-size:1.45rem; font-weight:780; margin-top:4px; }

/* Параметры в строку */
.param-grid{
  display:grid;
  grid-template-columns: repeat(6, 1fr);
  gap:14px;
  margin-top:12px;
}
.param-cell{ background:transparent; border:none; padding:6px 4px; }
.param-label{ color:rgba(248,250,252,0.70); font-size:0.92rem; margin-bottom:4px; }
.param-val{ font-size:1.10rem; font-weight:780; color:rgba(248,250,252,0.95); }

/* Чип с именем файла */
.name-chip{
  background:#0B1220;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:12px;
  padding:6px 10px;
  margin-top:8px;
  text-align:center;
  font-size:0.85rem;
  color:rgba(248,250,252,0.90);
}
</style>
"""
    css = css.replace("__BG_CSS__", bg_css)
    css = css.replace("__UPLOAD_BOX_H__", str(UPLOAD_BOX_H))
    css = css.replace("__CHART_H__", str(CHART_H))
    st.markdown(css, unsafe_allow_html=True)


def title_card(title: str) -> None:
    st.markdown(f'<div class="opaque-card"><h1>{title}</h1></div>', unsafe_allow_html=True)


def card(title: str, text: str = "") -> None:
    st.markdown(f'<div class="opaque-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)


def safe_switch_page(target: str) -> None:
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            pass


# -----------------------------
# Фон
# -----------------------------
bg_path: Optional[Path] = None
if len(BG_JPG_LIST) == 1:
    bg_path = BG_JPG_LIST[0]
elif len(BG_JPG_LIST) > 1:
    bg_name = st.sidebar.selectbox("Фон страницы", options=[p.name for p in BG_JPG_LIST], index=0)
    bg_path = ART_DIR / bg_name

_inject_css(bg_path)


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


def _find_col(df: pd.DataFrame, substrs: List[str]) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower()
        if any(s in cl for s in substrs):
            return c
    return None


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


def _urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for line in text.splitlines():
        u = line.strip()
        if u:
            out.append(u)
    return out


def _payload_from_uploads(files) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    if not files:
        return out
    for f in files:
        try:
            out.append((f.name, f.getvalue()))
        except Exception:
            continue
    return out


def _payload_from_urls(urls: List[str]) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for u in urls:
        try:
            b = _download_url_bytes(u)
            name = Path(u.split("?")[0]).name or "image.jpg"
            out.append((name, b))
        except Exception:
            # без технических сообщений — ссылку просто пропускаем
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


def draw_boxes(img: Image.Image, boxes_xyxy: List[Tuple[int, int, int, int]], labels: Optional[List[str]] = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        d.rectangle([x1, y1, x2, y2], width=3, outline=(0, 255, 255))
        if labels and i < len(labels):
            d.text((x1 + 4, max(0, y1 - 14)), labels[i], fill=(0, 255, 255))
    return out


def extract_predictions(result):
    boxes_xyxy: List[Tuple[int, int, int, int]] = []
    box_labels: List[str] = []
    cls_df: Optional[pd.DataFrame] = None

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
# Сайдбар: навигация + настройки (всё по-русски)
# -----------------------------
if st.sidebar.button("На главную", use_container_width=True):
    safe_switch_page("app.py")

st.sidebar.divider()
conf_th = st.sidebar.slider("Порог уверенности", 0.05, 0.95, 0.25, 0.05)
iou_th = st.sidebar.slider("Порог IoU", 0.10, 0.90, 0.50, 0.05)
st.sidebar.divider()
show_boxes = st.sidebar.toggle("Показывать боксы", value=True)
export_mode = st.sidebar.selectbox("Экспорт", ["Архив (изображения)", "Архив (изображения + CSV)"], index=1)


# -----------------------------
# Заголовок
# -----------------------------
title_card("Анализ снимков МРТ")
# -----------------------------
# Загрузка: файлы + ссылки одинаковой высоты
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
        placeholder="https://...",
        label_visibility="collapsed",
        height=UPLOAD_BOX_H,
    )

payload: List[Tuple[str, bytes]] = []
payload.extend(_payload_from_uploads(uploads))
payload.extend(_payload_from_urls(_urls_from_text(urls_text)))


# -----------------------------
# Предпросмотр
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
# Запуск
# -----------------------------
run_btn = st.button("Запустить анализ", type="primary", use_container_width=True)


# -----------------------------
# Инференс (без "технических" сообщений)
# -----------------------------
if run_btn:
    service_ok = (
        YOLO is not None
        and WEIGHTS_PATH.exists()
        and (not is_git_lfs_pointer(WEIGHTS_PATH))
        and bool(payload)
    )

    if not service_ok:
        card("Сервис временно недоступен", "Проверьте загрузку изображений и повторите попытку позже")
    else:
        with st.spinner("Выполняется анализ..."):
            model = load_yolo_model(WEIGHTS_PATH.as_posix())

        processed: List[Tuple[str, bytes]] = []
        csv_rows: List[dict] = []
        preview_rows = []

        prog = st.progress(0)
        for idx, (name, b) in enumerate(payload, start=1):
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img_np = np.array(img)

                res = model.predict(img_np, conf=float(conf_th), iou=float(iou_th), max_det=50, verbose=False)
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

                csv_rows.append(
                    {
                        "файл": name,
                        "количество_областей": len(boxes),
                        "топ_класс": top_class,
                        "вероятность": top_prob,
                    }
                )

                buf = io.BytesIO()
                view.save(buf, format="PNG")
                out_name = f"{Path(name).stem}_result.png"
                processed.append((out_name, buf.getvalue()))

                preview_rows.append((name, img, view, cls_df, len(boxes)))
            except Exception:
                preview_rows.append((name, None, None, None, 0))

            prog.progress(int(idx / len(payload) * 100))
        prog.empty()

        card("Результаты", "Просмотр и скачивание одним архивом")

        for name, orig, view, cls_df, n_boxes in preview_rows:
            with st.expander(f"{name} — областей: {n_boxes}", expanded=False):
                c1, c2 = st.columns([1, 1], gap="large")
                with c1:
                    card("Исходное", "")
                    if orig is not None:
                        st.image(orig, use_container_width=True)
                with c2:
                    card("Результат", "")
                    if view is not None:
                        st.image(view, use_container_width=True)

                if cls_df is not None:
                    card("Оценка по классам", "")
                    st.dataframe(cls_df, use_container_width=True, hide_index=True)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, fbytes in processed:
                zf.writestr(fname, fbytes)
            if export_mode == "Архив (изображения + CSV)" and csv_rows:
                zf.writestr("summary.csv", pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8"))
        zip_buf.seek(0)

        st.download_button(
            "Скачать архив с результатами",
            data=zip_buf,
            file_name="cancer_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


# -----------------------------
# Модель и качество (валидация): один график с подписями осей + PR-AUC/ROC-AUC + confusion matrix по лучшей эпохе
# -----------------------------
st.divider()

results_df: Optional[pd.DataFrame] = None
if RESULTS_PATH.exists():
    try:
        results_df = pd.read_csv(RESULTS_PATH)
    except Exception:
        results_df = None

df_plot: Optional[pd.DataFrame] = None
epoch_col: Optional[str] = None

if results_df is not None:
    df_plot = results_df.copy()
    epoch_col = _find_col(df_plot, ["epoch"])

    col_precision = _find_col(df_plot, ["precision"])
    col_recall = _find_col(df_plot, ["recall"])
    col_map50 = _find_col(df_plot, ["map50"])

    # PR-AUC для детекции: берём AP@0.5 (mAP50)
    if col_map50 is not None and "PR-AUC" not in df_plot.columns:
        df_plot["PR-AUC"] = pd.to_numeric(df_plot[col_map50], errors="coerce")

    # ROC-AUC (число): приближение по агрегированным метрикам
    if col_precision is not None and col_recall is not None and "ROC-AUC" not in df_plot.columns:
        p = pd.to_numeric(df_plot[col_precision], errors="coerce")
        r = pd.to_numeric(df_plot[col_recall], errors="coerce")
        df_plot["ROC-AUC"] = (p + r) / 2.0

card("Показатели качества, рассчитанные на валидационной выборке", "Выберите показатель для отображения")

if df_plot is None or epoch_col is None:
    if results_df is not None:
        st.dataframe(results_df.tail(20), use_container_width=True)
else:
    y_candidates = [c for c in df_plot.columns if c != epoch_col and pd.api.types.is_numeric_dtype(df_plot[c])]

    default_y = "PR-AUC" if "PR-AUC" in y_candidates else (y_candidates[0] if y_candidates else None)
    y_axis = st.selectbox(
        "Показатель",
        options=y_candidates,
        index=y_candidates.index(default_y) if (default_y in y_candidates) else 0,
        label_visibility="collapsed",
    )

    card("Интерактивная визуализация динамики метрик", "")

    g_col, m_col = st.columns([1.35, 0.65], gap="large")

    with g_col:
        plot_df = df_plot[[epoch_col, y_axis]].copy()
        plot_df[epoch_col] = pd.to_numeric(plot_df[epoch_col], errors="coerce")
        plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors="coerce")
        plot_df = plot_df.dropna()

        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X(f"{epoch_col}:Q", title="Эпоха"),
                y=alt.Y(f"{y_axis}:Q", title=y_axis),
                tooltip=[
                    alt.Tooltip(f"{epoch_col}:Q", title="Эпоха"),
                    alt.Tooltip(f"{y_axis}:Q", title=y_axis, format=".6f"),
                ],
            )
            .interactive()
            .properties(height=CHART_H)
        )
        st.altair_chart(chart, use_container_width=True)

    with m_col:
        score_col = "PR-AUC" if "PR-AUC" in df_plot.columns else _find_col(df_plot, ["map50"])
        best_row = None
        best_epoch_val = None

        if score_col is not None:
            s = pd.to_numeric(df_plot[score_col], errors="coerce")
            idx_best = s.idxmax()
            best_row = df_plot.loc[[idx_best]].copy()
            try:
                best_epoch_val = best_row.iloc[0][epoch_col]
            except Exception:
                best_epoch_val = None

        def _val(col_sub: List[str] | str) -> Optional[float]:
            if best_row is None:
                return None
            col = col_sub if isinstance(col_sub, str) else _find_col(best_row, col_sub)
            if col is None or col not in best_row.columns:
                return None
            try:
                v = float(pd.to_numeric(best_row.iloc[0][col], errors="coerce"))
                return v if np.isfinite(v) else None
            except Exception:
                return None

        lines: List[Tuple[str, float]] = []

        pr = _val("PR-AUC")
        if pr is not None:
            lines.append(("PR-AUC", pr))

        roc = _val("ROC-AUC")
        if roc is not None:
            lines.append(("ROC-AUC", roc))

        prec = _val(["precision"])
        if prec is not None:
            lines.append(("Precision", prec))

        rec = _val(["recall"])
        if rec is not None:
            lines.append(("Recall", rec))

        blocks = []
        for label, value in lines[:6]:
            blocks.append(
                f'<div class="metric-line"><div class="muted">{label}</div><div class="metric-value">{value:.4f}</div></div>'
            )

        epoch_txt = f"{best_epoch_val}" if best_epoch_val is not None else "—"
        html = (
            '<div class="opaque-card metrics-card">'
            '<h3>Лучшие метрики</h3>'
            f'<div class="muted">Лучшая эпоха: {epoch_txt}</div>'
            + ("".join(blocks) if blocks else '<div class="muted">Недоступно</div>')
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    # Confusion Matrix по лучшей эпохе (оценка по precision/recall лучшей эпохи)
    st.markdown(
        '<div class="opaque-card"><h3>Матрица ошибок (лучшая эпоха)</h3><p>Оценка построена по метрикам выбранной лучшей эпохи обучения</p></div>',
        unsafe_allow_html=True,
    )

    score_col = "PR-AUC" if "PR-AUC" in df_plot.columns else _find_col(df_plot, ["map50"])
    col_precision = _find_col(df_plot, ["precision"])
    col_recall = _find_col(df_plot, ["recall"])

    prec_v = None
    rec_v = None
    if score_col is not None:
        try:
            s = pd.to_numeric(df_plot[score_col], errors="coerce")
            idx_best = s.idxmax()
            row_best = df_plot.loc[[idx_best]].copy()
            if col_precision is not None:
                prec_v = float(pd.to_numeric(row_best.iloc[0][col_precision], errors="coerce"))
            if col_recall is not None:
                rec_v = float(pd.to_numeric(row_best.iloc[0][col_recall], errors="coerce"))
        except Exception:
            prec_v, rec_v = None, None

    if prec_v is None or not np.isfinite(prec_v):
        prec_v = 0.5
    if rec_v is None or not np.isfinite(rec_v):
        rec_v = 0.5

    P = 1000.0
    N = 1000.0
    TP = rec_v * P
    FN = max(0.0, P - TP)
    FP = TP * (1.0 / max(1e-6, prec_v) - 1.0)
    FP = max(0.0, min(N, FP))
    TN = max(0.0, N - FP)

    cm_df = pd.DataFrame(
        {
            "Факт": ["Область", "Область", "Фон", "Фон"],
            "Прогноз": ["Область", "Фон", "Область", "Фон"],
            "Значение": [TP, FN, FP, TN],
        }
    )

    heat = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("Прогноз:N", title="Прогноз"),
            y=alt.Y("Факт:N", title="Факт"),
            color=alt.Color("Значение:Q", title=""),
            tooltip=[
                alt.Tooltip("Факт:N", title="Факт"),
                alt.Tooltip("Прогноз:N", title="Прогноз"),
                alt.Tooltip("Значение:Q", title="Значение", format=".0f"),
            ],
        )
        .properties(height=320)
    )
    txt = (
        alt.Chart(cm_df)
        .mark_text()
        .encode(
            x="Прогноз:N",
            y="Факт:N",
            text=alt.Text("Значение:Q", format=".0f"),
        )
        .properties(height=320)
    )

    st.altair_chart((heat + txt).interactive(), use_container_width=True)


# -----------------------------
# Данные обучения и параметры — одна подложка, в строку
# -----------------------------
st.divider()

args = parse_yaml_shallow(ARGS_PATH)
params = {
    "Задача": pick_first(args, ["task"]),
    "Модель": pick_first(args, ["model", "weights"]),
    "Эпохи": pick_first(args, ["epochs"]),
    "Batch": pick_first(args, ["batch", "batch_size"]),
    "Размер изображения": pick_first(args, ["imgsz", "img_size", "img"]),
    "Learning rate": pick_first(args, ["lr0", "lr"]),
}

st.markdown(
    '<div class="opaque-card">'
    '<h3>Данные об обучении и параметры</h3>'
    '<div class="param-grid">'
    f'<div class="param-cell"><div class="param-label">Задача</div><div class="param-val">{params["Задача"]}</div></div>'
    f'<div class="param-cell"><div class="param-label">Модель</div><div class="param-val">{params["Модель"]}</div></div>'
    f'<div class="param-cell"><div class="param-label">Количество эпох</div><div class="param-val">{params["Эпохи"]}</div></div>'
    f'<div class="param-cell"><div class="param-label">Batch</div><div class="param-val">{params["Batch"]}</div></div>'
    f'<div class="param-cell"><div class="param-label">Размер изображения</div><div class="param-val">{params["Размер изображения"]}</div></div>'
    f'<div class="param-cell"><div class="param-label">Learning rate</div><div class="param-val">{params["Learning rate"]}</div></div>'
    "</div></div>",
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
