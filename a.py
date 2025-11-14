import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# دالة لحساب المسافة بالكيلومتر (معادلة هفرسين)
# ----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # نصف قطر الأرض بالكيلومتر

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ----------------------------
# قراءة ملف (CSV أو Excel)
# ----------------------------
def load_table(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

# ----------------------------
# تطبيق Streamlit
# ----------------------------
st.set_page_config(page_title="تحقق العدادات حول المواقع", layout="wide")
st.title("🔍 التحقق من وجود عدادات حول المساجد / المدارس")

st.markdown(
    """
ارفع:
- ملف **عدادات الإدارة** (إحداثيات العدادات)
- ملف **المواقع** (مساجد، مدارس، ... من Outscraper مثلاً)

وسيتم إرجاع جميع المواقع التي **لا يوجد ضمن نصف قطرها أي عداد**.
"""
)

# 1) رفع الملفات
st.header("1️⃣ رفع الملفات")

meters_file = st.file_uploader("📂 ملف عدادات الإدارة (CSV أو Excel)", type=["csv", "xlsx"], key="meters")
sites_file = st.file_uploader("📂 ملف المواقع (مساجد / مدارس)", type=["csv", "xlsx"], key="sites")

if meters_file is not None and sites_file is not None:
    meters_df = load_table(meters_file)
    sites_df = load_table(sites_file)

    st.subheader("معاينة سريعة")
    st.write("عدادات الإدارة:")
    st.dataframe(meters_df.head())
    st.write("المواقع:")
    st.dataframe(sites_df.head())

    # 2) اختيار أعمدة الإحداثيات والاسم
    st.header("2️⃣ تحديد الأعمدة المستخدمة")

    st.markdown("### عدادات الإدارة")
    meter_lat_col = st.selectbox(
        "عمود خط العرض (latitude) للعدادات",
        meters_df.columns,
        index=list(meters_df.columns).index("latitude") if "latitude" in meters_df.columns else 0,
    )
    meter_lon_col = st.selectbox(
        "عمود خط الطول (longitude) للعدادات",
        meters_df.columns,
        index=list(meters_df.columns).index("longitude") if "longitude" in meters_df.columns else 0,
    )

    st.markdown("### المواقع")
    site_lat_col = st.selectbox(
        "عمود خط العرض (latitude) للمواقع",
        sites_df.columns,
        index=list(sites_df.columns).index("latitude") if "latitude" in sites_df.columns else 0,
    )
    site_lon_col = st.selectbox(
        "عمود خط الطول (longitude) للمواقع",
        sites_df.columns,
        index=list(sites_df.columns).index("longitude") if "longitude" in sites_df.columns else 0,
    )

    # عمود اسم الموقع للعرض (في ملف المساجد من Outscraper العمود اسمه name)
    site_name_col = st.selectbox(
        "عمود اسم الموقع (اختياري ولكن يُفضّل)",
        sites_df.columns,
        index=list(sites_df.columns).index("name") if "name" in sites_df.columns else 0,
    )

    # 3) نصف القطر
    st.header("3️⃣ إعدادات النطاق")
    radius_km = st.number_input(
        "نصف القطر حول كل موقع (كم)",
        min_value=0.1,
        max_value=50.0,
        value=0.5,
        step=0.1,
    )

    if st.button("▶ تنفيذ التحقق"):
        # تنظيف وتحويل الأعمدة الرقمية
        meters_coords = meters_df[[meter_lat_col, meter_lon_col]].apply(pd.to_numeric, errors="coerce").dropna()
        sites_coords = sites_df[[site_lat_col, site_lon_col]].apply(pd.to_numeric, errors="coerce").dropna()

        if meters_coords.empty:
            st.error("لا توجد إحداثيات صالحة في ملف العدادات بعد التنظيف.")
        elif sites_coords.empty:
            st.error("لا توجد إحداثيات صالحة في ملف المواقع بعد التنظيف.")
        else:
            # نربط الإحداثيات النظيفة بالصفوف الأصلية بحسب الفهرس
            sites_valid = sites_df.loc[sites_coords.index].copy()
            sites_valid["__lat"] = sites_coords[site_lat_col].values
            sites_valid["__lon"] = sites_coords[site_lon_col].values

            meter_lats = meters_coords[meter_lat_col].values
            meter_lons = meters_coords[meter_lon_col].values

            # دالة لحساب أقل مسافة لموقع واحد إلى جميع العدادات
            def compute_min_distance(row):
                dists = haversine_distance(row["__lat"], row["__lon"], meter_lats, meter_lons)
                return float(np.min(dists))

            st.info("جاري حساب أقل مسافة من كل موقع إلى أقرب عداد، قد يستغرق ذلك قليلاً حسب حجم البيانات...")
            sites_valid["min_distance_km"] = sites_valid.apply(compute_min_distance, axis=1)
            sites_valid["has_meter_in_radius"] = sites_valid["min_distance_km"] <= radius_km

            no_meter_df = sites_valid[~sites_valid["has_meter_in_radius"]].copy()

            st.header("4️⃣ النتائج")

            st.write(f"إجمالي عدد المواقع (صحيحة الإحداثيات): **{len(sites_valid)}**")
            st.write(f"عدد المواقع التي يوجد عداد ضمن نصف القطر: **{int(sites_valid['has_meter_in_radius'].sum())}**")
            st.write(f"عدد المواقع التي لا يوجد عداد ضمن نصف القطر: **{len(no_meter_df)}**")

            st.subheader("📌 المواقع بدون عدادات ضمن النطاق المحدد")
            cols_to_show = []
            if site_name_col in no_meter_df.columns:
                cols_to_show.append(site_name_col)
            if site_lat_col in no_meter_df.columns:
                cols_to_show.append(site_lat_col)
            if site_lon_col in no_meter_df.columns:
                cols_to_show.append(site_lon_col)
            cols_to_show.append("min_distance_km")

            st.dataframe(no_meter_df[cols_to_show])

            # تجهيز ملف التحميل
            out_csv = no_meter_df.drop(columns=["__lat", "__lon"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇ تحميل ملف المواقع بدون عدادات (CSV)",
                data=out_csv,
                file_name="sites_without_meters.csv",
                mime="text/csv",
            )

else:
    st.info("⬆ رجاءً ارفع ملف العدادات وملف المواقع للبدء.")
