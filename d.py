# -*- coding: utf-8 -*-
import os
import io
import math
import time
import base64
import calendar
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
from collections import defaultdict

# ==========================
# إعدادات عامة
# ==========================

CHANGE_THRESHOLD = 0.15         # عتبة درجة التغيّر لاعتبار الموقع "نشط"
OUTPUT_IMG_DIR = "output_images"

# أسماء الأعمدة في ملف العدادات (كما في ملفك)
COL_OFFICE       = "المكتب"
COL_METER_ID     = "التجهيزات"
COL_NAME         = "الاسم"
COL_SUBSCRIPTION = "الاشتراك"
COL_CATEGORY     = "الفئة"
COL_LON          = "longitude"
COL_LAT          = "latitude"
COL_PLACE        = "مكان"

# إعدادات CDSE
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# 🔍 حجم المشهد حول العداد (قلّلناه أكثر لتقريب شديد على الموقع)
SCENE_SIZE_M = 500        # جرّب 250 أو حتى 150 لو تبغى أقوى تقريب
IMG_SIZE_PX  = 640        # حجم الصورة


# ==========================
# تحميل ملف العدادات
# ==========================

def load_meters_excel(file) -> pd.DataFrame:
    """قراءة ملف العدادات من Excel وتوحيد أسماء الأعمدة."""
    df = pd.read_excel(file, dtype={COL_METER_ID: str, COL_SUBSCRIPTION: str})

    df = df.rename(columns={
        COL_OFFICE:       "office",
        COL_METER_ID:     "meter_id",
        COL_NAME:         "customer_name",
        COL_SUBSCRIPTION: "subscription",
        COL_CATEGORY:     "category",
        COL_LON:          "longitude",
        COL_LAT:          "latitude",
        COL_PLACE:        "place_code"
    })

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    return df


# ==========================
# NDVI (تجريبي حالياً)
# ==========================

def fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date):
    """NDVI تجريبي شهري بين تاريخين (استبدله لاحقاً بالحقيقي)."""
    months = pd.date_range(start_date, end_date, freq="MS")  # بداية كل شهر
    if len(months) == 0:
        return months, np.array([])

    base = np.random.uniform(0.2, 0.6)
    noise = np.random.normal(0, 0.05, size=len(months))
    trend = np.linspace(-0.1, 0.1, len(months))
    ndvi_values = np.clip(base + trend + noise, 0.0, 1.0)
    return months, ndvi_values


def compute_change_score_for_meter(lat, lon, start_date, end_date):
    """حساب درجة تغيّر NDVI بين أول وآخر شهر."""
    months, ndvi_values = fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date)

    if len(ndvi_values) < 2:
        change_score = 0.0
    else:
        change_score = float(abs(ndvi_values[-1] - ndvi_values[0]))

    return change_score, months, ndvi_values


def classify_status(change_score, threshold=CHANGE_THRESHOLD):
    """تصنيف مبدئي للنشاط."""
    if change_score >= threshold:
        return "نشط", "✅"
    else:
        return "مهجور محتمل", "⚠️"


# ==========================
# إدارة المجلدات + حفظ NDVI
# ==========================

def ensure_output_dir():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def save_ndvi_plot(meter_id, months, ndvi_values):
    """حفظ منحنى NDVI في مجلد العداد."""
    ensure_output_dir()
    meter_folder = os.path.join(OUTPUT_IMG_DIR, str(meter_id))
    os.makedirs(meter_folder, exist_ok=True)

    plt.figure(figsize=(4, 3))
    plt.plot(months, ndvi_values, marker="o")
    plt.title(f"منحنى NDVI للعداد {meter_id}")
    plt.xlabel("التاريخ")
    plt.ylabel("NDVI")
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(meter_folder, "ndvi_timeseries.png")
    plt.savefig(img_path)
    plt.close()
    return img_path


# ==========================
# دوال CDSE (Catalog + Process)
# ==========================

def bbox_from_meters(lat: float, lon: float, size_m: float):
    half = size_m / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def get_cdse_token():
    """الحصول على توكن CDSE وتخزينه في session_state."""
    tok = st.session_state.get("_cdse_token")
    exp = st.session_state.get("_cdse_token_exp", 0)
    if tok and time.time() < exp - 60:
        return tok

    cid = st.secrets.get("CDSE_CLIENT_ID")
    csec = st.secrets.get("CDSE_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET غير موجودة في secrets")

    data = {
        "grant_type": "client_credentials",
        "client_id": cid,
        "client_secret": csec
    }
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"CDSE token error {r.status_code}: {r.text[:200]}")

    js = r.json()
    access = js["access_token"]
    expires = int(js.get("expires_in", 3600))
    st.session_state["_cdse_token"] = access
    st.session_state["_cdse_token_exp"] = time.time() + expires
    return access


@st.cache_data(show_spinner=False, ttl=24*3600)
def get_month_s2_dates(lat: float, lon: float, year: int, month: int, max_items: int = 20):
    """جلب تواريخ مشاهد Sentinel-2 فوق الموقع خلال شهر معيّن."""
    token = get_cdse_token()
    bbox = bbox_from_meters(lat, lon, SCENE_SIZE_M)
    last_day = calendar.monthrange(year, month)[1]
    dt_range = f"{year}-{month:02d}-01T00:00:00Z/{year}-{month:02d}-{last_day:02d}T23:59:59Z"

    payload = {
        "bbox": bbox,
        "collections": ["sentinel-2-l2a"],
        "datetime": dt_range,
        "limit": max_items
    }

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(CATALOG_URL, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        st.warning(f"Catalog status {r.status_code}: {r.text[:200]}")
        return []

    js = r.json()
    feats = js.get("features", [])
    dates = set()
    for f in feats:
        props = f.get("properties", {})
        dt_str = props.get("datetime") or props.get("date") or ""
        if "T" in dt_str:
            dt_str = dt_str.split("T")[0]
        if dt_str:
            dates.add(dt_str)

    return sorted(list(dates))


@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat: float, lon: float, meter_id: str,
                   acq_date: str,
                   timeout: int = 30):
    """
    تنزيل مشهد Sentinel-2 True Color حول العداد،
    مع رسم علامة حمراء واضحة في مركز الصورة (موقع العداد).
    """
    ensure_output_dir()
    meter_folder = os.path.join(OUTPUT_IMG_DIR, str(meter_id))
    os.makedirs(meter_folder, exist_ok=True)

    img_path = os.path.join(meter_folder, f"site_{acq_date}.png")
    if os.path.exists(img_path):
        return img_path

    bbox = bbox_from_meters(lat, lon, SCENE_SIZE_M)

    def _request(token):
        data_filter = {
            "maxCloudCoverage": 60,
            "mosaickingOrder": "mostRecent",
            "timeRange": {
                "from": f"{acq_date}T00:00:00Z",
                "to":   f"{acq_date}T23:59:59Z"
            }
        }
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": data_filter,
                    "processing": {"upsampling": "NEAREST", "downsampling": "NEAREST"}
                }]
            },
            "output": {
                "width": IMG_SIZE_PX,
                "height": IMG_SIZE_PX,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/png"}
                }]
            },
            "evalscript": """//VERSION=3
function setup(){return {input:["B04","B03","B02"],output:{bands:3}}}
function evaluatePixel(s){
  return [s.B04*1.8, s.B03*1.8, s.B02*1.8]
}
"""
        }
        headers = {"Authorization": f"Bearer " + token}
        return requests.post(PROCESS_URL, headers=headers, json=payload, timeout=timeout)

    token = get_cdse_token()
    r = _request(token)
    if r.status_code == 401:
        token = get_cdse_token()
        r = _request(token)

    if r.status_code == 200:
        img_bytes = io.BytesIO(r.content)
        img = Image.open(img_bytes).convert("RGB")

        # 🔴 علامة أوضح: + سميكة + دائرة حمراء في المركز
        draw = ImageDraw.Draw(img)
        cx, cy = IMG_SIZE_PX // 2, IMG_SIZE_PX // 2
        line_len = 26      # طول أذرع علامة +
        line_w   = 5       # سماكة الخط
        color    = (255, 0, 0)

        # خطوط +
        draw.line([(cx - line_len, cy), (cx + line_len, cy)], fill=color, width=line_w)
        draw.line([(cx, cy - line_len), (cx, cy + line_len)], fill=color, width=line_w)

        # دائرة صغيرة في المركز
        radius = 8
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            outline=color, width=3
        )

        img.save(img_path)
        return img_path
    else:
        st.warning(f"Copernicus status {r.status_code} للعداد {meter_id} ({acq_date}): {r.text[:200]}")
        return None


# ==========================
# حفظ Excel + HTML
# ==========================

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


def save_results_html(results_rows, gallery, start_date: date, end_date: date) -> bytes:
    """تقرير HTML: بطاقة + صور لكل عداد."""

    def border_color(status: str) -> str:
        return "#4CAF50" if status == "نشط" else "#ff9800"

    period_str = f"{start_date} – {end_date}"

    html = [
        "<html><head><meta charset='UTF-8'>",
        "<title>تقرير حالات العدادات</title>",
        "<style>",
        "body{font-family:Tahoma,Arial,sans-serif;direction:rtl;text-align:right;background:#f7f7f7;margin:0;padding:10px;}",
        ".container{max-width:960px;margin:0 auto;}",
        ".card{background:#fff;border-radius:10px;padding:12px;margin:10px 0;border:3px solid #ccc;",
        "  page-break-inside: avoid; break-inside: avoid; -webkit-column-break-inside: avoid;}",
        ".thumbs{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;}",
        ".thumb{border:1px solid #ddd;border-radius:8px;padding:4px;background:#fafafa;text-align:center;",
        "  page-break-inside: avoid; break-inside: avoid; -webkit-column-break-inside: avoid;width:180px;}",
        ".thumb img{border-radius:6px;max-width:170px;height:auto;}",
        "h2{margin-top:0;}",
        "@media print {",
        "  body{background:#fff;padding:0;}",
        "  .container{max-width:100%;margin:0 10mm;}",
        "  .card{border:1px solid #000;margin:5mm 0;padding:5mm;font-size:11pt;}",
        "  .thumb{width:45%;margin-bottom:4mm;}",
        "  img{max-width:100%;height:auto;}",
        "}",
        "</style>",
        "</head><body>",
        "<div class='container'>",
        f"<h2>تقرير حالات العدادات للفترة: {period_str}</h2>"
    ]

    for row in results_rows:
        meter_id = row["meter_id"]
        status = row["status"]
        change_score = row["change_score"]
        office = row.get("office", "")
        cat = row.get("category", "")
        sub = row.get("subscription", "")
        lat = row["latitude"]
        lon = row["longitude"]
        change_pct = change_score * 100.0
        imgs = gallery.get(meter_id, [])
        border = border_color(status)

        best_img_tag = ""
        if imgs:
            imgs_sorted = sorted(imgs, key=lambda x: x["date"])
            best_info = None
            for inf in imgs_sorted[::-1]:
                if "قمر" in inf.get("label", ""):
                    best_info = inf
                    break
            if best_info is None:
                best_info = imgs_sorted[-1]
            if os.path.exists(best_info["img_path"]):
                with open(best_info["img_path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                best_img_tag = (
                    f"<img src='data:image/png;base64,{b64}' "
                    f"width='280' style='border-radius:8px;border:2px solid #333;'>"
                )

        html.append(f"<div class='card' style='border-color:{border};'>")
        html.append(f"<h3>عداد {meter_id} ({status})</h3>")
        if best_img_tag:
            html.append(f"<div>{best_img_tag}</div>")

        html.append(
            f"<p>"
            f"<strong>درجة التغيّر:</strong> {change_score:.3f} ({change_pct:.1f}%) &nbsp; | "
            f"<strong>المكتب:</strong> {office} &nbsp; | "
            f"<strong>الفئة:</strong> {cat}<br>"
            f"<strong>رقم الاشتراك:</strong> {sub}<br>"
            f"<strong>الإحداثيات:</strong> Lat {lat:.6f}, Lon {lon:.6f} &nbsp; "
            f"<a href='https://maps.google.com?q={lat},{lon}'>رابط الموقع على الخريطة</a>"
            f"</p>"
        )

        if imgs:
            html.append("<h4>صور التغيّر (مرتّبة زمنيًا):</h4>")
            html.append("<div class='thumbs'>")
            for info in sorted(imgs, key=lambda x: x["date"]):
                if not os.path.exists(info["img_path"]):
                    continue
                with open(info["img_path"], "rb") as f:
                    g64 = base64.b64encode(f.read()).decode()
                d = info["date"]
                if isinstance(d, (pd.Timestamp, datetime)):
                    d_str = d.strftime("%Y-%m-%d")
                elif isinstance(d, date):
                    d_str = d.strftime("%Y-%m-%d")
                else:
                    d_str = str(d)
                label = info.get("label", "صورة")
                html.append(
                    "<div class='thumb'>"
                    f"<img src='data:image/png;base64,{g64}'><br>"
                    f"<small>{label}<br>التاريخ: {d_str}</small>"
                    "</div>"
                )
            html.append("</div>")

        html.append("</div>")  # card

    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")


# ==========================
# واجهة Streamlit
# ==========================

def main():
    st.set_page_config(
        page_title="تحليل نشاط العدادات من صور الأقمار الصناعية",
        page_icon="📡",
        layout="wide"
    )

    st.title("📡 نظام تقدير نشاط العدادات باستخدام صور الأقمار الصناعية")

    with st.sidebar:
        st.header("الإعدادات")

        today = date.today()
        start_date = st.date_input(
            "تاريخ البداية",
            value=date(today.year, 1, 1),
            key="meters_start_date"
        )
        end_date = st.date_input(
            "تاريخ النهاية",
            value=today,
            key="meters_end_date"
        )

        st.markdown("---")
        st.write("مصدر الصور:")
        st.markdown("- **CDSE Sentinel-2 True Color** (نفس نظام المزارع).")

        st.markdown("---")
        st.write(f"سيتم حفظ صور كل عداد في المجلد: `{OUTPUT_IMG_DIR}/<رقم_العداد>/`")

    uploaded_file = st.file_uploader("📁 ارفع ملف العدادات (Excel)", type=["xlsx", "xls"])

    if uploaded_file is None:
        st.info("الرجاء رفع ملف العدادات للبدء.")
        return

    try:
        meters_df = load_meters_excel(uploaded_file)
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {e}")
        return

    st.success(f"تم تحميل الملف، عدد العدادات: {len(meters_df)}")

    if st.checkbox("👀 عرض أول 10 سجلات من الملف"):
        st.dataframe(meters_df.head(10))

    if not st.button("🚀 بدء التحليل"):
        return

    st.write("### النتائج (تظهر الحالات هنا واحدة تلو الأخرى أثناء التحليل)")
    cards_container = st.container()

    results_rows = []
    gallery = defaultdict(list)

    n = len(meters_df)
    progress = st.progress(0)
    t0 = time.time()

    cols = cards_container.columns(3)
    col_idx = 0

    for i, (_, row) in enumerate(meters_df.iterrows(), start=1):
        try:
            meter_id = row["meter_id"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            # 1) NDVI ودرجة التغيّر
            change_score, months, ndvi_values = compute_change_score_for_meter(
                lat, lon, start_date, end_date
            )
            status, icon = classify_status(change_score)
            change_pct = round(change_score * 100, 1)

            # 2) حفظ منحنى NDVI + إضافته للـ gallery
            if len(months) > 0 and len(ndvi_values) == len(months):
                ndvi_plot_path = save_ndvi_plot(meter_id, months, ndvi_values)
                gallery[meter_id].append({
                    "label": "منحنى NDVI",
                    "date": months[0],
                    "img_path": ndvi_plot_path,
                })

            # 3) صور القمر الصناعي (مشهد واحد لكل شهر في الفترة)
            months_range = pd.date_range(start_date, end_date, freq="MS")
            for m_dt in months_range:
                year = int(m_dt.year)
                month = int(m_dt.month)
                dates_for_month = get_month_s2_dates(lat, lon, year, month)
                if not dates_for_month:
                    continue

                acq_date = dates_for_month[0]  # أول مشهد في الشهر
                img_path = download_image(lat, lon, meter_id, acq_date)
                if img_path is None:
                    continue

                gallery[meter_id].append({
                    "label": "صورة قمر صناعي",
                    "date": pd.to_datetime(acq_date),
                    "img_path": img_path,
                })

            # 4) نسجّل النتيجة
            results_rows.append({
                "meter_id": meter_id,
                "office": row.get("office"),
                "subscription": row.get("subscription"),
                "category": row.get("category"),
                "place_code": row.get("place_code"),
                "latitude": lat,
                "longitude": lon,
                "change_score": round(change_score, 3),
                "status": status,
                "status_icon": icon,
            })

            # 5) عرض البطاقة والصور
            imgs_for_meter = sorted(gallery[meter_id], key=lambda x: x["date"])
            main_img_path = None
            for inf in imgs_for_meter[::-1]:
                if "قمر" in inf.get("label", ""):
                    main_img_path = inf["img_path"]
                    break
            if main_img_path is None and imgs_for_meter:
                main_img_path = imgs_for_meter[-1]["img_path"]

            with cols[col_idx % 3]:
                st.markdown(
                    f"<div style='border:2px solid #ccc;border-radius:10px;"
                    f"padding:8px;margin:6px;text-align:center;font-size:13px;'>"
                    f"<strong>عداد {meter_id}</strong><br>"
                    f"{icon} {status} | Δ={change_score:.3f} ({change_pct}%)<br>"
                    f"مكتب: {row.get('office','')} | فئة: {row.get('category','')}<br>"
                    f"<a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if main_img_path and os.path.exists(main_img_path):
                    st.image(
                        main_img_path,
                        caption="أحدث مشهد (العلامة الحمراء = موقع العداد)",
                        width=220
                    )

                with st.expander("📂 جميع صور هذا العداد"):
                    for info in imgs_for_meter:
                        if not os.path.exists(info["img_path"]):
                            continue
                        d = info["date"]
                        if isinstance(d, (pd.Timestamp, datetime)):
                            d_str = d.strftime("%Y-%m-%d")
                        elif isinstance(d, date):
                            d_str = d.strftime("%Y-%m-%d")
                        else:
                            d_str = str(d)
                        st.image(
                            info["img_path"],
                            caption=f"{info['label']} | التاريخ: {d_str}",
                            width=200
                        )

            col_idx += 1
            progress.progress(i / max(n, 1))

        except Exception as e:
            st.warning(f"⚠️ خطأ في العداد {row.get('subscription','?')}: {e}")
            progress.progress(i / max(n, 1))
            continue

    st.success(f"✅ اكتمل التحليل في {time.time() - t0:.1f} ثانية")

    if not results_rows:
        st.warning("لا توجد نتائج لتحميلها.")
        return

    results_df = pd.DataFrame(results_rows)

    total_meters = len(results_df)
    active_count = int((results_df["status"] == "نشط").sum())
    inactive_count = int((results_df["status"] == "مهجور محتمل").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("إجمالي العدادات", total_meters)
    c2.metric("عدادات نشطة (مبدئيًا)", active_count)
    c3.metric("عدادات مهجورة محتملة", inactive_count)

    st.markdown("---")

    excel_bytes = to_excel_bytes(results_df)
    st.download_button(
        label="📥 تحميل جدول النتائج (Excel)",
        data=excel_bytes,
        file_name=f"meters_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    html_bytes = save_results_html(results_rows, gallery, start_date, end_date)
    st.download_button(
        label="📄 تحميل تقرير HTML مع الصور",
        data=html_bytes,
        file_name=f"meters_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )


if __name__ == "__main__":
    main()
