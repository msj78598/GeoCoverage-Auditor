import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import geopandas as gpd

# لو حاب تربط SentinelHub، فعل السطور التالية وعدّل الإعدادات في الدالة الخاصة به:
# from sentinelhub import SHConfig, SentinelHubStatistical, DataCollection, Geometry

# ==========================
# إعدادات عامة للنظام
# ==========================

USE_DUMMY_DATA = True        # True = تحليل تجريبي بدون أقمار صناعية
CHANGE_THRESHOLD = 0.15      # عتبة التغيير لتصنيف الموقع نشط

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
    # نقرأ الملف مع التأكد من أن أرقام العدادات كنص
    df = pd.read_excel(file, dtype={COL_METER_ID: str, COL_SUBSCRIPTION: str})

    # نعيد تسمية الأعمدة لأسماء موحدة
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
    ndvi_values = np.random.uniform(0.1, 0.8, size=len(months))
    return months, ndvi_values


# مثال ربط مع SentinelHub (تعدّلها حسب إعداداتك):
"""
def fetch_ndvi_timeseries_real(lat, lon, start_date, end_date):
    config = SHConfig()
    config.instance_id = "YOUR_INSTANCE_ID"
    config.sh_client_id = "YOUR_CLIENT_ID"
    config.sh_client_secret = "YOUR_CLIENT_SECRET"

    # نقطة صغيرة حول العداد
    point_geom = Geometry(
        {"type": "Point", "coordinates": [lon, lat]},
        crs="EPSG:4326"
    )

    time_interval = (start_date.isoformat(), end_date.isoformat())

    request = SentinelHubStatistical(
        aggregation=...,
        input_data=[...],
        geometry=point_geom,
        config=config
    )

    stats = request.get_data()
    # حوّل stats إلى سلسلة زمنية NDVI (تواريخ + قيم)
    months = ...
    ndvi_values = ...
    return months, ndvi_values
"""


def compute_change_score_for_meter(lat, lon, start_date, end_date):
    """
    يحسب درجة التغيّر لموقع واحد بين تاريخين.
    - يرجع:
        change_score: رقم بين 0 و 1
        months: قائمة تواريخ
        ndvi_values: قيم NDVI لكل شهر (لأغراض العرض فقط)
    """
    if USE_DUMMY_DATA:
        months, ndvi_values = fetch_ndvi_timeseries_dummy(lat, lon, start_date, end_date)
    else:
        # استبدلها بدالتك الحقيقية:
        # months, ndvi_values = fetch_ndvi_timeseries_real(lat, lon, start_date, end_date)
        raise NotImplementedError("اربط دالة fetch_ndvi_timeseries_real قبل تعطيل USE_DUMMY_DATA")

    # حساب درجة التغيّر: الفرق بين أول وآخر قيمة NDVI
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


def analyze_meters(gdf: gpd.GeoDataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    يمر على جميع العدادات ويحسب درجة التغيّر والحالة لكل واحد.
    يرجع DataFrame بالنتائج.
    """
    results = []

    for idx, row in gdf.iterrows():
        meter_id = row["meter_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        change_score, months, ndvi_values = compute_change_score_for_meter(lat, lon, start_date, end_date)
        status, icon = classify_status(change_score)

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
            "status_icon": icon
        })

    result_df = pd.DataFrame(results)
    return result_df


# ==========================
# واجهة المستخدم (Streamlit)
# ==========================

def main():
    st.set_page_config(page_title="تحليل نشاط العدادات من صور الأقمار الصناعية", layout="wide")

    st.title("نظام تحليل نشاط العدادات باستخدام صور الأقمار الصناعية")
    st.write("ارفع ملف العدادات (إكسل) وسيقوم النظام بتقدير ما إذا كانت المواقع نشطة أو مهجورة اعتمادًا على التغيّر الزمني.")

    with st.sidebar:
        st.header("الإعدادات")

        start_date = st.date_input("تاريخ البداية", value=date(date.today().year, 1, 1))
        end_date = st.date_input("تاريخ النهاية", value=date.today())

        st.markdown("---")
        st.write("وضع البيانات:")
        st.write("- **تجريبي** إذا كان `USE_DUMMY_DATA = True` في الكود.")
        st.write("- غيّرها إلى False واربط دالة الأقمار الصناعية لنتائج حقيقية.")

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
            with st.spinner("جاري تحليل العدادات..."):
                results_df = analyze_meters(gdf, start_date, end_date)

            st.success("اكتمل التحليل")

            # عرض ملخص
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي العدادات", len(results_df))
            col2.metric("العدادات النشطة (مبدئيًا)", (results_df["status"] == "نشط").sum())
            col3.metric("العدادات المهجورة المحتملة", (results_df["status"] == "مهجور محتمل").sum())

            st.subheader("نتائج تفصيلية")
            st.dataframe(results_df)

            # تحميل النتائج كإكسل
            excel_bytes = to_excel_bytes(results_df)
            st.download_button(
                label="تحميل النتائج كملف Excel",
                data=excel_bytes,
                file_name=f"meters_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.info("الرجاء رفع ملف العدادات للبدء.")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    يحول DataFrame إلى ملف Excel في الذاكرة لإتاحته للتحميل من Streamlit.
    """
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


if __name__ == "__main__":
    main()
