# -*- coding: utf-8 -*-
import os
import math
import time
import calendar
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests

# ==========================
# إعدادات عامة
# ==========================

USE_DUMMY_DATA = False          # الآن نستخدم بيانات صور حقيقية من CDSE
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

# إعدادات CDSE (نفس نظام المزارع)
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

SCENE_SIZE_M = 2500       # حجم المشهد بالمتر (مثل نظام المزارع)
IMG_SIZE_PX  = 640        # حجم الصورة الناتجة

# ==========================
# تحميل ملف العدادات
# ==========================

def load_meters_excel(file) -> pd.DataFrame:
    """
    يقرأ ملف العدادات من Excel ويعيد DataFrame مع أعمدة موحدة الأسماء.
    """
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
    """
    دالة تجريبية ترجع سلسلة زمنية شهرية لـ NDVI بين تاريخين.
    الهدف فقط اختبار النظام؛ استبدلها لاحقاً بدالتك الحقيقية.
    """
    months = pd.date_range(start_date, end_date, freq="MS")  # بداية كل شهر
    if len(months) == 0:
        return months, np.array([])

    base = np.random.uniform(0.2, 0.6)
    noise = np.random.normal(0, 0.05, size=len(months))
    trend = np.linspace(-0.1, 0.1, len(months))
    ndvi_values = np.clip(base + trend + noise, 0.0, 1.0)
    return months, ndvi_values


def compute_change_score_for_meter(lat, lon, start_date, end_date):
    """
    يحسب درجة التغيّر لموقع واحد بين تاريخين:
      - change_score: فرق NDVI بين أول وآخر شهر (0–1 تقريباً)
      - months: قائمة تواريخ
      - ndvi_values: قيم NDVI لكل شهر
    """
    # حالياً NDVI تجريبي – يمكنك لاحقاً ربطه بـ SentinelHub Statistical API
    months, ndvi_values = fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date)

    if len(ndvi_values) < 2:
        change_score = 0.0
    else:
        change_score = float(abs(ndvi_values[-1] - ndvi_values[0]))

    return change_score, months, ndvi_values


def classify_status(change_score, threshold=CHANGE_THRESHOLD):
    """
    تصنيف مبدئي للموقع بناءً على درجة التغيّر.
    """
    if change_score >= threshold:
        return "نشط", "✅"
    else:
        return "مهجور محتمل", "⚠️"


# ==========================
# إدارة المجلدات
# ==========================

def ensure_output_dir():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def save_ndvi_plot(meter_id, months, ndvi_values):
    """
    يحفظ منحنى NDVI للعداد في صورة PNG داخل مجلد العداد.
    """
    ensure_output_dir()
    meter_folder = os.path.join(OUTPUT_IMG_DIR, str(meter_id))
    os.makedirs(meter_folder, exist_ok=True)

    plt.figure()
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
# دوال CDSE (مأخوذة ومبسّطة من نظام المزارع)
# ==========================

def bbox_from_meters(lat: float, lon: float, size_m: float):
    half = size_m / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def get_cdse_token():
    """
    نفس فكرة نظام المزارع: نخزن التوكن في session_state ونجدده عند الحاجة.
    """
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
    """
    ترجع قائمة تواريخ (YYYY-MM-DD) لمشاهد Sentinel-2 فوق الموقع خلال شهر معين.
    نفس منطق نظام المزارع.
    """
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


def download_image(lat: float, lon: float, meter_id: str,
                   acq_date: str,
                   timeout: int = 30):
    """
    تنزيل مشهد Sentinel-2 True Color بنفس طريقة نظام المزارع،
    وحفظه داخل مجلد العداد.
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
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path
    else:
        st.warning(f"Copernicus status {r.status_code} للعداد {meter_id} ({acq_date}): {r.text[:200]}")
        return None


# ==========================
# تحليل العدادات + بناء جدول النتائج + مجلد الصور
# ==========================

from collections import defaultdict

def analyze_meters(df: pd.DataFrame, start_date: date, end_date: date):
    """
    يمر على كل عداد:
      - يحسب NDVI ودرجة التغيّر (تجريبي الآن)
      - يجلب صورة قمر صناعي واحدة لكل شهر بين التاريخين
    يرجع:
      - results_df: جدول الحالات
      - gallery: dict  meter_id -> list of {label, date, img_path}
    """
    results = []
    gallery = defaultdict(list)

    for _, row in df.iterrows():
        meter_id = row["meter_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        # 1) NDVI ودرجة التغيّر
        change_score, months, ndvi_values = compute_change_score_for_meter(
            lat, lon, start_date, end_date
        )
        status, icon = classify_status(change_score)

        # 2) حفظ منحنى NDVI
        if len(months) > 0 and len(ndvi_values) == len(months):
            ndvi_plot_path = save_ndvi_plot(meter_id, months, ndvi_values)
            gallery[meter_id].append({
                "label": "منحنى NDVI",
                "date": months[0],    # نربط المنحنى بأول شهر
                "img_path": ndvi_plot_path,
            })

        # 3) صور القمر الصناعي لكل شهر (باستخدام نفس طريقة نظام المزارع)
        # نبني أشهر الفترة
        months_range = pd.date_range(start_date, end_date, freq="MS")
        for m_dt in months_range:
            year = int(m_dt.year)
            month = int(m_dt.month)

            # نجيب كل تواريخ Sentinel-2 في هذا الشهر
            dates_for_month = get_month_s2_dates(lat, lon, year, month)
            if not dates_for_month:
                continue

            # نختار أول تاريخ (ممكن تغيره للمنتصف أو آخر الشهر)
            acq_date = dates_for_month[0]   # "YYYY-MM-DD"
            img_path = download_image(lat, lon, meter_id, acq_date)
            if img_path is None:
                continue

            gallery[meter_id].append({
                "label": "صورة قمر صناعي",
                "date": pd.to_datetime(acq_date),
                "img_path": img_path,
            })

        results.append({
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

    results_df = pd.DataFrame(results)
    return results_df, gallery


# ==========================
# أدوات الإكسل
# ==========================

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


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

    # إعدادات جانبية
    with st.sidebar:
        st.header("الإعدادات")

        today = date.today()
        start_date = st.date_input("تاريخ البداية", value=date(today.year, 1, 1))
        end_date = st.date_input("تاريخ النهاية", value=today)

        st.markdown("---")
        st.write("مصدر الصور:")
        st.markdown("- **CDSE Sentinel-2 True Color** (نفس نظام المزارع).")

        st.markdown("---")
        st.write(f"سيتم حفظ صور كل عداد في المجلد: `{OUTPUT_IMG_DIR}/<رقم_العداد>/`")

    uploaded_file = st.file_uploader("📁 ارفع ملف العدادات (Excel)", type=["xlsx", "xls"])

    if uploaded_file is None:
        st.info("الرجاء رفع ملف العدادات للبدء.")
        return

    # تحميل الملف
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

    # تشغيل التحليل
    with st.spinner("جاري تحليل العدادات وجلب صور الأقمار الصناعية..."):
        results_df, gallery = analyze_meters(meters_df, start_date, end_date)

    st.success("✅ اكتمل التحليل")

    # ملخص أعلى الصفحة
    total_meters = len(results_df)
    active_count = int((results_df["status"] == "نشط").sum())
    inactive_count = int((results_df["status"] == "مهجور محتمل").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("إجمالي العدادات", total_meters)
    c2.metric("عدادات نشطة (مبدئيًا)", active_count)
    c3.metric("عدادات مهجورة محتملة", inactive_count)

    st.markdown("---")
    st.subheader("📊 جدول الحالات مع صور التغيّر")

    # ====== عرض كل حالة: صف تفاصيل + صف صور ======
    for _, row in results_df.iterrows():
        meter_id = row["meter_id"]
        status   = row["status"]
        icon     = row["status_icon"]
        score    = row["change_score"]
        office   = row.get("office", "")
        cat      = row.get("category", "")
        sub      = row.get("subscription", "")
        lat      = row["latitude"]
        lon      = row["longitude"]

        change_pct = round(score * 100, 1)

        # --- الصف الأول: تفاصيل الحالة في "شكل جدول" ---
        c1, c2, c3, c4, c5, c6 = st.columns([1.6, 1.4, 1.0, 1.0, 1.0, 1.4])

        c1.markdown(
            f"**رقم العداد:** {meter_id}<br>"
            f"**رقم الاشتراك:** {sub}",
            unsafe_allow_html=True
        )
        c2.markdown(
            f"**الحالة:** {icon} {status}",
            unsafe_allow_html=True
        )
        c3.markdown(
            f"**درجة التغيّر:** {score} ({change_pct}%)",
            unsafe_allow_html=True
        )
        c4.markdown(
            f"**المكتب:** {office}",
            unsafe_allow_html=True
        )
        c5.markdown(
            f"**الفئة:** {cat}",
            unsafe_allow_html=True
        )
        c6.markdown(
            f"[📍 الموقع](https://maps.google.com?q={lat},{lon})<br>"
            f"Lat: {lat:.6f}<br>Lon: {lon:.6f}",
            unsafe_allow_html=True
        )

        # --- الصف الثاني: صور القمر الصناعي لهذا العداد ---
        imgs = gallery.get(meter_id, [])
        if imgs:
            imgs_sorted = sorted(imgs, key=lambda x: x["date"])
            n_per_row = 3
            num_imgs = len(imgs_sorted)
            rows = math.ceil(num_imgs / n_per_row)
            idx = 0

            for r in range(rows):
                cols = st.columns(n_per_row)
                for c in range(n_per_row):
                    if idx >= num_imgs:
                        break
                    info = imgs_sorted[idx]
                    idx += 1

                    img_path = info["img_path"]
                    if not os.path.exists(img_path):
                        continue

                    date_val = info["date"]
                    if isinstance(date_val, (pd.Timestamp, datetime)):
                        date_str = date_val.strftime("%Y-%m-%d")
                    elif isinstance(date_val, date):
                        date_str = date_val.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_val)

                    label = info.get("label", "صورة")
                    with cols[c]:
                        st.image(
                            img_path,
                            caption=f"{label} | التاريخ: {date_str}",
                            use_column_width=True
                        )
        else:
            st.info("لا توجد صور محفوظة لهذا العداد (قد لا تتوفر مشاهد في الأشهر المحددة).")

        st.markdown("---")

    # زر تحميل النتائج كإكسل
    excel_bytes = to_excel_bytes(results_df)
    st.download_button(
        label="📥 تحميل جدول النتائج (Excel)",
        data=excel_bytes,
        file_name=f"meters_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()
