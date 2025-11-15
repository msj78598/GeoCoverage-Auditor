import os
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# ==========================
# إعدادات عامة
# ==========================

USE_DUMMY_DATA = True          # غيّرها إلى False عندما تربط الدوال الحقيقية
CHANGE_THRESHOLD = 0.15
OUTPUT_IMG_DIR = "output_images"

# أسماء الأعمدة في ملف العدادات
COL_OFFICE       = "المكتب"
COL_METER_ID     = "التجهيزات"
COL_NAME         = "الاسم"
COL_SUBSCRIPTION = "الاشتراك"
COL_CATEGORY     = "الفئة"
COL_LON          = "longitude"
COL_LAT          = "latitude"
COL_PLACE        = "مكان"


# ==========================
# قراءة ملف العدادات
# ==========================

def load_meters_excel(file) -> gpd.GeoDataFrame:
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

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    return gdf


# ==========================
# دوال الأقمار الصناعية (NDVI + صور RGB)
# ==========================

def fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date):
    months = pd.date_range(start_date, end_date, freq="MS")
    base = np.random.uniform(0.2, 0.6)
    noise = np.random.normal(0, 0.05, size=len(months))
    trend = np.linspace(-0.1, 0.1, len(months))
    ndvi_values = np.clip(base + trend + noise, 0.0, 1.0)
    return months, ndvi_values


def fetch_rgb_image_dummy(lat, lon, on_date):
    """صورة تجريبية (رمادية) – استبدلها لاحقًا بدالتك الحقيقية من CDSE."""
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    return img

# لو حاب تربط صور حقيقية:
# def fetch_rgb_image_real(lat, lon, on_date):
#     ...
#     return pil_image


def compute_change_score_for_meter(lat, lon, start_date, end_date):
    if USE_DUMMY_DATA:
        months, ndvi_values = fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date)
    else:
        # استبدل بالنداء الحقيقي بعد ربط دالتك
        # months, ndvi_values = fetch_ndvi_timeseries_real(lat, lon, start_date, end_date)
        raise NotImplementedError("اربط دالة NDVI الحقيقية ثم غيّر USE_DUMMY_DATA إلى False")

    if len(ndvi_values) < 2:
        change_score = 0.0
    else:
        change_score = float(abs(ndvi_values[-1] - ndvi_values[0]))

    return change_score, months, ndvi_values


def classify_status(change_score, threshold=CHANGE_THRESHOLD):
    if change_score >= threshold:
        return "نشط", "✅"
    else:
        return "مهجور محتمل", "⚠️"


# ==========================
# إدارة المجلدات وحفظ الصور
# ==========================

def ensure_output_dirs():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)


def save_ndvi_plot(meter_id, months, ndvi_values):
    ensure_output_dirs()
    meter_folder = os.path.join(OUTPUT_IMG_DIR, str(meter_id))
    os.makedirs(meter_folder, exist_ok=True)

    plt.figure()
    plt.plot(months, ndvi_values, marker="o")
    plt.title(f"NDVI Timeseries - Meter {meter_id}")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(meter_folder, "ndvi_timeseries.png")
    plt.savefig(img_path)
    plt.close()
    return img_path


def save_rgb_snapshots(meter_id, lat, lon, start_date, end_date):
    """
    يحفظ صورتين (بداية ونهاية الفترة) لكل عداد.
    ترجع المسارات لاستخدامها لاحقًا في العرض.
    لو أضفت صور أكثر (شهرية مثلاً) إلى نفس المجلد،
    واجهة المجلد ستعرضها كلها تلقائياً.
    """
    ensure_output_dirs()
    meter_folder = os.path.join(OUTPUT_IMG_DIR, str(meter_id))
    os.makedirs(meter_folder, exist_ok=True)

    if USE_DUMMY_DATA:
        img_start = fetch_rgb_image_dummy(lat, lon, start_date)
        img_end = fetch_rgb_image_dummy(lat, lon, end_date)
    else:
        # img_start = fetch_rgb_image_real(lat, lon, start_date)
        # img_end   = fetch_rgb_image_real(lat, lon, end_date)
        raise NotImplementedError("اربط دالة الصور الحقيقية ثم غيّر USE_DUMMY_DATA إلى False")

    start_path = os.path.join(meter_folder, "site_start.png")
    end_path   = os.path.join(meter_folder, "site_end.png")

    img_start.save(start_path)
    img_end.save(end_path)

    return start_path, end_path


def analyze_meters(gdf: gpd.GeoDataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    results = []

    for idx, row in gdf.iterrows():
        meter_id = row["meter_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        change_score, months, ndvi_values = compute_change_score_for_meter(lat, lon, start_date, end_date)
        status, icon = classify_status(change_score)

        ndvi_plot_path = save_ndvi_plot(meter_id, months, ndvi_values)
        site_start_path, site_end_path = save_rgb_snapshots(meter_id, lat, lon, start_date, end_date)

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
            "ndvi_plot_path": ndvi_plot_path,
            "site_start_path": site_start_path,
            "site_end_path": site_end_path,
        })

    return pd.DataFrame(results)


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
    st.set_page_config(page_title="تحليل نشاط العدادات من صور الأقمار الصناعية", layout="wide")

    st.title("تحليل نشاط العدادات باستخدام صور الأقمار الصناعية")

    if "open_meter_id" not in st.session_state:
        st.session_state["open_meter_id"] = None

    with st.sidebar:
        st.header("الإعدادات")

        start_date = st.date_input("تاريخ البداية", value=date(date.today().year, 1, 1))
        end_date = st.date_input("تاريخ النهاية", value=date.today())

        st.markdown("---")
        st.write("وضع البيانات:")
        if USE_DUMMY_DATA:
            st.markdown("- **تجريبي**: NDVI وصور الموقع افتراضية (لاختبار النظام).")
        else:
            st.markdown("- **حقيقي**: يعتمد على دوال Copernicus/CDSE التي تربطها.")

        st.markdown("---")
        st.write(f"الصور تُحفظ في المجلد: `{OUTPUT_IMG_DIR}/<meter_id>/`")

    uploaded_file = st.file_uploader("ارفع ملف العدادات (xlsx / xls)", type=["xlsx", "xls"])

    if uploaded_file is None:
        st.info("الرجاء رفع ملف العدادات للبدء.")
        return

    try:
        gdf = load_meters_excel(uploaded_file)
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {e}")
        return

    st.success(f"تم تحميل الملف، عدد العدادات: {len(gdf)}")

    if st.checkbox("عرض أول 10 سجلات من الملف"):
        st.dataframe(gdf.head(10))

    if not st.button("بدء التحليل"):
        return

    with st.spinner("جاري تحليل العدادات وحفظ الصور..."):
        results_df = analyze_meters(gdf, start_date, end_date)

    st.success("اكتمل التحليل")

    c1, c2, c3 = st.columns(3)
    c1.metric("إجمالي العدادات", len(results_df))
    c2.metric("العدادات النشطة", int((results_df["status"] == "نشط").sum()))
    c3.metric("المهجورة المحتملة", int((results_df["status"] == "مهجور محتمل").sum()))

    # ========= جدول النتائج مع أيقونة مجلد =========
    st.subheader("جدول النتائج")
    st.write("اضغط على أيقونة المجلّد 📁 لعرض مجلد صور موقع العداد وتفسير النتيجة بصريًا.")

    # عناوين الأعمدة
    header_cols = st.columns([1.3, 1.2, 0.8, 1.0, 1.0, 0.6])
    header_cols[0].markdown("**رقم العداد**")
    header_cols[1].markdown("**الحالة**")
    header_cols[2].markdown("**درجة التغيّر**")
    header_cols[3].markdown("**المكتب**")
    header_cols[4].markdown("**الفئة**")
    header_cols[5].markdown("**📁**")

    st.markdown("---")

    for idx, row in results_df.iterrows():
        cols = st.columns([1.3, 1.2, 0.8, 1.0, 1.0, 0.6])
        cols[0].write(str(row["meter_id"]))
        cols[1].write(f"{row['status_icon']} {row['status']}")
        cols[2].write(row["change_score"])
        cols[3].write(str(row.get("office", "")))
        cols[4].write(str(row.get("category", "")))

        open_folder = cols[5].button("📁", key=f"open_{idx}")

        if open_folder:
            st.session_state["open_meter_id"] = row["meter_id"]

    # ====== مجلد صور العداد المختار ======
    open_id = st.session_state.get("open_meter_id")
    if open_id is not None:
        st.markdown("---")
        st.subheader(f"📁 مجلد صور موقع العداد {open_id}")

        meter_folder = os.path.join(OUTPUT_IMG_DIR, str(open_id))
        if os.path.exists(meter_folder):
            image_files = [
                f for f in os.listdir(meter_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            image_files.sort()

            if not image_files:
                st.warning("لا توجد صور محفوظة لهذا العداد في المجلد.")
            else:
                for f in image_files:
                    img_path = os.path.join(meter_folder, f)
                    st.image(img_path, caption=f)
        else:
            st.warning("لم يتم العثور على مجلد لهذا العداد (تحقق من مسار الحفظ).")

    # زر تحميل إكسل في الأسفل
    st.markdown("---")
    excel_bytes = to_excel_bytes(results_df)
    st.download_button(
        label="تحميل النتائج كملف Excel",
        data=excel_bytes,
        file_name=f"meters_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()
