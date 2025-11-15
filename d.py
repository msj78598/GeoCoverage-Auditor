import os
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================
# إعدادات عامة للنظام
# ==========================

# True = يستخدم بيانات NDVI عشوائية للتجربة (الكود يشتغل فوراً)
# عندما تربط دالة الأقمار الصناعية الحقيقية غيّرها إلى False
USE_DUMMY_DATA = True

# عتبة التغيّر لتصنيف الموقع "نشط"
CHANGE_THRESHOLD = 0.15

# مجلد حفظ صور النتائج لكل عداد
OUTPUT_IMG_DIR = "output_images"

# أسماء الأعمدة في ملف الإكسل (كما في الصورة)
COL_OFFICE       = "المكتب"
COL_METER_ID     = "التجهيزات"
COL_NAME         = "الاسم"
COL_SUBSCRIPTION = "الاشتراك"
COL_CATEGORY     = "الفئة"
COL_LON          = "longitude"
COL_LAT          = "latitude"
COL_PLACE        = "مكان"      # كود المكان (اختياري)


# ==========================
# دوال مساعدة على قراءة البيانات
# ==========================

def load_meters_excel(file) -> gpd.GeoDataFrame:
    """
    يقرأ ملف العدادات من Streamlit uploader ويعيده كـ GeoDataFrame
    بنفس أسلوب مشروع الفاقد في الفئة الزراعية.
    """
    df = pd.read_excel(file, dtype={COL_METER_ID: str, COL_SUBSCRIPTION: str})

    # إعادة تسمية الأعمدة لأسماء موحدة
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

    # تنظيف الإحداثيات
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    # بناء GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    return gdf


# ==========================
# دوال الأقمار الصناعية
# ==========================

def fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date):
    """
    دالة تجريبية (بدون أقمار صناعية) تعطي NDVI عشوائي لكل شهر.
    الهدف منها اختبار النظام والواجهة فقط.
    """
    months = pd.date_range(start_date, end_date, freq="MS")  # بداية كل شهر
    # نجعل القيم فيها بعض التغيّر التدريجي عشان تكون منطقية أكثر
    base = np.random.uniform(0.2, 0.6)
    noise = np.random.normal(0, 0.05, size=len(months))
    trend = np.linspace(-0.1, 0.1, len(months))
    ndvi_values = np.clip(base + trend + noise, 0.0, 1.0)
    return months, ndvi_values


# ======= مكان ربط كود الأقمار الصناعية الحقيقي =======
# هنا تحط نفس الكود اللي استخدمته في مشروع الفاقد الزراعي
# لجلب NDVI شهري لنقطة (lat, lon).
# مثلاً باستخدام Copernicus / SentinelHub / Google Earth Engine.
#
# مثال هيكل دالة (عدّلها حسب كودك):
#
# def fetch_ndvi_timeseries_real(lat, lon, start_date, end_date):
#     """
#     TODO: اربطها بكودك الحقيقي (Copernicus / Sentinel).
#     لازم ترجع:
#       months: list of datetime
#       ndvi_values: numpy array of float
#     """
#     # استخدم st.secrets["CDSE_CLIENT_ID"] , st.secrets["CDSE_CLIENT_SECRET"]
#     # أو غيرها من المتغيرات التي حفظتها في Secrets
#     raise NotImplementedError("اربط هذه الدالة بكود الأقمار الصناعية الحقيقي.")
# =====================================================


def compute_change_score_for_meter(lat, lon, start_date, end_date):
    """
    يحسب درجة التغيّر لموقع واحد بين تاريخين.
    - يرجع:
        change_score: رقم بين 0 و 1 تقريباً
        months: قائمة تواريخ
        ndvi_values: قيم NDVI لكل شهر
    """
    if USE_DUMMY_DATA:
        months, ndvi_values = fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date)
    else:
        # استبدل هذا باستدعاء دالتك الحقيقية:
        # months, ndvi_values = fetch_ndvi_timeseries_real(lat, lon, start_date, end_date)
        raise NotImplementedError("غيّر USE_DUMMY_DATA إلى True أو اربط الدالة الحقيقية.")

    if len(ndvi_values) < 2:
        change_score = 0.0
    else:
        change_score = float(abs(ndvi_values[-1] - ndvi_values[0]))

    return change_score, months, ndvi_values


def classify_status(change_score, threshold=CHANGE_THRESHOLD):
    """
    تصنيف الموقع بناءً على درجة التغيّر.
    """
    if change_score >= threshold:
        return "نشط", "✅"
    else:
        return "مهجور محتمل", "⚠️"


# ==========================
# حفظ الصور في مجلد لكل عداد
# ==========================

def ensure_output_dirs():
    """إنشاء مجلد رئيسي لحفظ الصور إذا لم يكن موجوداً."""
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)


def save_ndvi_plot(meter_id, months, ndvi_values):
    """
    يحفظ شكل منحنى NDVI لكل عداد داخل مجلد خاص:
    output_images/<meter_id>/ndvi_timeseries.png
    """
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


def analyze_meters(gdf: gpd.GeoDataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    يمر على جميع العدادات ويحسب درجة التغيّر والحالة لكل واحد.
    كما يحفظ صورة منحنى NDVI لكل عداد في مجلد مستقل.
    يرجع DataFrame بالنتائج.
    """
    results = []

    for idx, row in gdf.iterrows():
        meter_id = row["meter_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        change_score, months, ndvi_values = compute_change_score_for_meter(lat, lon, start_date, end_date)
        status, icon = classify_status(change_score)

        # حفظ صورة منحنى NDVI لهذا العداد
        img_path = save_ndvi_plot(meter_id, months, ndvi_values)

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
            "ndvi_plot_path": img_path
        })

    result_df = pd.DataFrame(results)
    return result_df


# ==========================
# أدوات مساعدة للإكسل
# ==========================

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    يحول DataFrame إلى ملف Excel في الذاكرة لإتاحته للتحميل من Streamlit.
    """
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


# ==========================
# واجهة المستخدم (Streamlit)
# ==========================

def main():
    st.set_page_config(page_title="تحليل نشاط العدادات من صور الأقمار الصناعية", layout="wide")

    st.title("نظام تحليل نشاط العدادات باستخدام صور الأقمار الصناعية")
    st.write(
        "هذا النظام يقرأ ملف العدادات، يحسب تغيّر NDVI لكل موقع عبر الزمن، "
        "ويقدّر ما إذا كان الموقع نشطًا أو مهجورًا، مع حفظ صورة لكل حالة في مجلد مستقل."
    )

    with st.sidebar:
        st.header("الإعدادات")

        start_date = st.date_input("تاريخ البداية", value=date(date.today().year, 1, 1))
        end_date = st.date_input("تاريخ النهاية", value=date.today())

        st.markdown("---")
        st.write("وضع البيانات:")
        if USE_DUMMY_DATA:
            st.markdown("- **تجريبي**: يستخدم NDVI عشوائي للاختبار.")
        else:
            st.markdown("- **حقيقي**: يتطلب ربط دالة الأقمار الصناعية.")

        st.markdown("---")
        st.write(f"سيتم حفظ صور النتائج في المجلد: `{OUTPUT_IMG_DIR}/<meter_id>/`")

    uploaded_file = st.file_uploader("ارفع ملف العدادات (xlsx / xls)", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # قراءة الملف
        try:
            gdf = load_meters_excel(uploaded_file)
        except Exception as e:
            st.error(f"خطأ في قراءة الملف: {e}")
            return

        st.success(f"تم تحميل الملف، عدد العدادات: {len(gdf)}")

        if st.checkbox("عرض أول 10 سجلات من الملف"):
            st.dataframe(gdf.head(10))

        if st.button("بدء التحليل"):
            with st.spinner("جاري تحليل العدادات وحفظ الصور..."):
                results_df = analyze_meters(gdf, start_date, end_date)

            st.success("اكتمل التحليل")

            # عرض ملخص
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي العدادات", len(results_df))
            col2.metric("العدادات النشطة (مبدئيًا)", int((results_df["status"] == "نشط").sum()))
            col3.metric("العدادات المهجورة المحتملة", int((results_df["status"] == "مهجور محتمل").sum()))

            st.subheader("نتائج تفصيلية")
            st.dataframe(results_df[[
                "status_icon", "status", "change_score",
                "meter_id", "office", "category",
                "subscription", "place_code",
                "latitude", "longitude"
            ]])

            # تحميل النتائج كإكسل
            excel_bytes = to_excel_bytes(results_df)
            st.download_button(
                label="تحميل النتائج كملف Excel",
                data=excel_bytes,
                file_name=f"meters_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # اختيار عداد لعرض صورته
            st.markdown("---")
            st.subheader("عرض بصري لنتيجة عدّاد معيّن")

            sel_meter = st.selectbox(
                "اختر عدّاد لعرض منحنى NDVI (تم حفظه أيضًا في المجلد):",
                results_df["meter_id"].astype(str).tolist()
            )

            sel_row = results_df[results_df["meter_id"].astype(str) == str(sel_meter)].iloc[0]
            st.write(f"الحالة: {sel_row['status_icon']} {sel_row['status']} | درجة التغيّر: {sel_row['change_score']}")

            img_path = sel_row["ndvi_plot_path"]
            if os.path.exists(img_path):
                st.image(img_path, caption=f"منحنى NDVI للعداد {sel_meter}")
            else:
                st.warning("لم يتم العثور على صورة لهذا العداد (تحقق من المجلد على السيرفر).")

    else:
        st.info("الرجاء رفع ملف العدادات للبدء.")


if __name__ == "__main__":
    main()
