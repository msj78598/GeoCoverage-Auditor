# -*- coding: utf-8 -*-
import os
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# ==========================
# إعدادات عامة
# ==========================

USE_DUMMY_DATA = True          # غيّرها إلى False لما تربط دوال الأقمار الصناعية الحقيقية
CHANGE_THRESHOLD = 0.15        # عتبة درجة التغير لاعتبار الموقع "نشط"
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
# دوال NDVI وصور الأقمار (تجريبية الآن)
# ==========================

def fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date):
    """
    دالة تجريبية ترجع سلسلة زمنية شهرية لـ NDVI بين تاريخين.
    الهدف فقط اختبار النظام؛ استبدلها لاحقاً بدالتك الحقيقية.
    """
    months = pd.date_range(start_date, end_date, freq="MS")  # بداية كل شهر
    base = np.random.uniform(0.2, 0.6)
    noise = np.random.normal(0, 0.05, size=len(months))
    trend = np.linspace(-0.1, 0.1, len(months))
    ndvi_values = np.clip(base + trend + noise, 0.0, 1.0)
    return months, ndvi_values


def fetch_rgb_image_dummy(lat, lon, on_date):
    """
    صورة تجريبية (مربّع رمادي) – استبدلها لاحقاً بدالة تجلب صورة من CDSE/Sentinel.
    """
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    return img


def compute_change_score_for_meter(lat, lon, start_date, end_date):
    """
    يحسب درجة التغيّر لموقع واحد بين تاريخين:
      - change_score: فرق NDVI بين أول وآخر شهر (0–1 تقريباً)
      - months: قائمة تواريخ
      - ndvi_values: قيم NDVI لكل شهر
    """
    if USE_DUMMY_DATA:
        months, ndvi_values = fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date)
    else:
        # هنا تستدعي دالتك الحقيقية لجلب NDVI
        # months, ndvi_values = fetch_ndvi_timeseries_real(lat, lon, start_date, end_date)
        raise NotImplementedError("اربط دالة NDVI الحقيقية ثم غيّر USE_DUMMY_DATA إلى False")

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
# إدارة المجلدات وحفظ الصور
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


def save_site_image(meter_id, on_date, img_pil):
    """
    يحفظ صورة موقع العداد لتاريخ معين داخل مجلد العداد.
    """
    ensure_output_dir()
    meter_folder = os.path.join(OUTPUT_IMG_DIR, str(meter_id))
    os.makedirs(meter_folder, exist_ok=True)

    date_str = on_date.strftime("%Y-%m-%d")
    img_name = f"site_{date_str}.png"
    img_path = os.path.join(meter_folder, img_name)
    img_pil.save(img_path)
    return img_path


# ==========================
# تحليل جميع العدادات + بناء جدول النتائج + مجلد الصور
# ==========================

from collections import defaultdict

def analyze_meters(df: pd.DataFrame, start_date: date, end_date: date):
    """
    يمر على كل عداد:
      - يحسب NDVI ودرجة التغيّر
      - يبني مجلد صور لكل عداد (منحنى NDVI + صور لكل شهر)
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

        change_score, months, ndvi_values = compute_change_score_for_meter(
            lat, lon, start_date, end_date
        )
        status, icon = classify_status(change_score)

        # 1) منحنى NDVI
        ndvi_plot_path = save_ndvi_plot(meter_id, months, ndvi_values)
        # نعتبر تاريخ المنحنى هو بداية الفترة لأغراض الترتيب
        if len(months) > 0:
            ndvi_date = months[0]
        else:
            ndvi_date = pd.to_datetime(start_date)

        gallery[meter_id].append({
            "label": "منحنى NDVI",
            "date": ndvi_date,
            "img_path": ndvi_plot_path,
        })

        # 2) صور لكل شهر في الفترة (تاريخ واضح + ترتيب زمني)
        for dt in months:
            on_date = dt.to_pydatetime().date()
            if USE_DUMMY_DATA:
                img_pil = fetch_rgb_image_dummy(lat, lon, on_date)
            else:
                # img_pil = fetch_rgb_image_real(lat, lon, on_date)
                raise NotImplementedError("اربط دالة صور القمر الصناعي الحقيقية ثم غيّر USE_DUMMY_DATA إلى False")

            img_path = save_site_image(meter_id, on_date, img_pil)
            gallery[meter_id].append({
                "label": "صورة قمر صناعي",
                "date": dt,
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
        st.write("وضع البيانات:")
        if USE_DUMMY_DATA:
            st.markdown("- **تجريبي**: NDVI وصور المواقع عشوائية (للاختبار فقط).")
        else:
            st.markdown("- **حقيقي**: يعتمد على دوال الأقمار الصناعية التي تربطها أنت.")

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
    with st.spinner("جاري تحليل العدادات وحفظ الصور..."):
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
            # ترتيب الصور حسب التاريخ
            imgs_sorted = sorted(imgs, key=lambda x: x["date"])

            # نعرضها في صفوف، كل صف فيه 3 صور
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
            st.info("لا توجد صور محفوظة لهذا العداد.")

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
