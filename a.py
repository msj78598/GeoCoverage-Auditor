import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# دالة لحساب المسافة بالكيلومتر (هفرسين)
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
st.set_page_config(page_title="GeoMeterX - Coverage Check", layout="wide")
st.title("🔍 التحقق من وجود عدادات حول المساجد / المدارس")

st.markdown(
    """
ارفع:
- ملف **عدادات الإدارة** (إحداثيات العدادات وكل بياناتها)
- ملف **المواقع** (مساجد، مدارس، ... من Outscraper مثلاً)

سيتم:
- تحديد المواقع التي **لا يوجد ضمن نصف قطرها أي عداد**.
- إظهار تفاصيل **أقرب عداد** لكل موقع يوجد ضمن نطاقه عداد.
- إتاحة تحميل النتائج كملفات Excel/CSV.
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

    # 2) اختيار أعمدة الإحداثيات والاسم والرابط
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

    st.markdown("### المواقع (مساجد / مدارس)")
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

    site_name_col = st.selectbox(
        "عمود اسم الموقع (اختياري)",
        sites_df.columns,
        index=list(sites_df.columns).index("name") if "name" in sites_df.columns else 0,
    )

    # عمود رابط خرائط جوجل (إن وجد)
    default_url_idx = list(sites_df.columns).index("url") if "url" in sites_df.columns else 0
    site_url_col = st.selectbox(
        "عمود رابط الموقع (Google Maps URL) - اختياري",
        sites_df.columns,
        index=default_url_idx,
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
        # تنظيف الإحداثيات
        meters_coords = meters_df[[meter_lat_col, meter_lon_col]].apply(pd.to_numeric, errors="coerce")
        meters_coords = meters_coords.dropna()
        meters_valid = meters_df.loc[meters_coords.index].copy()

        sites_coords = sites_df[[site_lat_col, site_lon_col]].apply(pd.to_numeric, errors="coerce")
        sites_coords = sites_coords.dropna()
        sites_valid = sites_df.loc[sites_coords.index].copy()

        if meters_valid.empty:
            st.error("لا توجد إحداثيات صالحة في ملف العدادات بعد التنظيف.")
        elif sites_valid.empty:
            st.error("لا توجد إحداثيات صالحة في ملف المواقع بعد التنظيف.")
        else:
            # ربط الإحداثيات النظيفة
            sites_valid["__lat"] = sites_coords[site_lat_col].values
            sites_valid["__lon"] = sites_coords[site_lon_col].values

            meter_details = meters_valid.reset_index(drop=True)
            meter_lats = meter_details[meter_lat_col].to_numpy()
            meter_lons = meter_details[meter_lon_col].to_numpy()

            # نحسب أقل مسافة + مؤشر أقرب عداد
            def compute_min_and_idx(row):
                dists = haversine_distance(row["__lat"], row["__lon"], meter_lats, meter_lons)
                idx = int(np.argmin(dists))
                return pd.Series({"min_distance_km": float(dists[idx]), "nearest_meter_idx": idx})

            st.info("جاري حساب أقرب عداد لكل موقع، قد يستغرق الأمر قليلاً حسب حجم البيانات...")
            tmp = sites_valid.apply(compute_min_and_idx, axis=1)
            sites_valid["min_distance_km"] = tmp["min_distance_km"]
            sites_valid["nearest_meter_idx"] = tmp["nearest_meter_idx"].astype(int)
            sites_valid["has_meter_in_radius"] = sites_valid["min_distance_km"] <= radius_km

            # إضافة تفاصيل أقرب عداد لكل موقع
            for col in meter_details.columns:
                sites_valid[f"meter_{col}"] = meter_details.iloc[
                    sites_valid["nearest_meter_idx"].values
                ][col].values

            # إضافة عمود رابط وصول (للاكسل + العرض)
            if site_url_col:
                def make_hyperlink(url):
                    if pd.isna(url) or url == "":
                        return ""
                    # في Excel ستكون خلية فيها =HYPERLINK("url","🔗 Open")
                    return f'=HYPERLINK("{url}", "🔗 Open")'
                sites_valid["maps_link"] = sites_valid[site_url_col].apply(make_hyperlink)

            # تقسيم النتائج
            with_meter_df = sites_valid[sites_valid["has_meter_in_radius"]].copy()
            no_meter_df = sites_valid[~sites_valid["has_meter_in_radius"]].copy()

            # عرض ملخص
            st.header("4️⃣ النتائج")

            st.write(f"إجمالي عدد المواقع (بإحداثيات صحيحة): **{len(sites_valid)}**")
            st.write(f"عدد المواقع التي يوجد عداد ضمن نصف القطر: **{len(with_meter_df)}**")
            st.write(f"عدد المواقع التي لا يوجد عداد ضمن نصف القطر: **{len(no_meter_df)}**")

            # أعمدة للعرض المختصر
            st.subheader("📌 المواقع بدون عدادات ضمن النطاق المحدد")
            cols_no_meter = []
            if site_name_col in no_meter_df.columns:
                cols_no_meter.append(site_name_col)
            cols_no_meter += [site_lat_col, site_lon_col, "min_distance_km"]
            if "maps_link" in no_meter_df.columns:
                cols_no_meter.append("maps_link")

            st.dataframe(no_meter_df[cols_no_meter])

            st.subheader("📌 المواقع التي يوجد ضمن نطاقها عداد + تفاصيل أقرب عداد")
            cols_with_meter = []
            if site_name_col in with_meter_df.columns:
                cols_with_meter.append(site_name_col)
            cols_with_meter += [site_lat_col, site_lon_col, "min_distance_km"]
            if "maps_link" in with_meter_df.columns:
                cols_with_meter.append("maps_link")

            # إضافة بعض أعمدة العداد المهمة أولاً (يمكنك تعديلها لاحقاً حسب ملفك)
            meter_main_cols = [c for c in meter_details.columns]  # كل الأعمدة
            cols_with_meter += [f"meter_{c}" for c in meter_main_cols]

            st.dataframe(with_meter_df[cols_with_meter])

            # تجهيز ملفات التحميل (CSV يمكن فتحه في Excel، والروابط تتحول لصيغة HYPERLINK)
            no_meter_csv = no_meter_df.drop(columns=["__lat", "__lon"], errors="ignore").to_csv(
                index=False, encoding="utf-8-sig"
            )
            with_meter_csv = with_meter_df.drop(columns=["__lat", "__lon"], errors="ignore").to_csv(
                index=False, encoding="utf-8-sig"
            )

            st.download_button(
                label="⬇ تحميل المواقع بدون عدادات (CSV/Excel)",
                data=no_meter_csv,
                file_name="sites_without_meters.csv",
                mime="text/csv",
            )

            st.download_button(
                label="⬇ تحميل المواقع مع أقرب عداد وتفاصيله (CSV/Excel)",
                data=with_meter_csv,
                file_name="sites_with_nearest_meters.csv",
                mime="text/csv",
            )

else:
    st.info("⬆ رجاءً ارفع ملف العدادات وملف المواقع للبدء.")
