import streamlit as st
import pandas as pd
import requests
import time
from io import BytesIO

# ----------------------------
# إعدادات عامة
# ----------------------------
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

st.set_page_config(
    page_title="تحليل العدادات المجمدة",
    layout="wide"
)

st.title("🔌 تحليل العدادات المجمدة باستخدام بيانات الخرائط (OSM)")
st.write("ارفع ملف العدادات المجمّدة (يحتوي على إحداثيات latitude / longitude وسبب التجميد)، وسيتم فحص حالة الموقع من الخريطة.")


# ----------------------------
# دوال مساعدة
# ----------------------------

def call_overpass(lat, lon, radius, overpass_url=DEFAULT_OVERPASS_URL):
    """
    استعلام بسيط عن أقرب نقطة نشاط حول العداد (amenity / building / shop / office)
    ضمن نصف قطر معيّن (متر).
    """
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius}, {lat}, {lon})["amenity"];
      way(around:{radius}, {lat}, {lon})["amenity"];
      relation(around:{radius}, {lat}, {lon})["amenity"];

      node(around:{radius}, {lat}, {lon})["building"];
      way(around:{radius}, {lat}, {lon})["building"];
      relation(around:{radius}, {lat}, {lon})["building"];

      node(around:{radius}, {lat}, {lon})["shop"];
      way(around:{radius}, {lat}, {lon})["shop"];
      relation(around:{radius}, {lat}, {lon})["shop"];

      node(around:{radius}, {lat}, {lon})["office"];
      way(around:{radius}, {lat}, {lon})["office"];
      relation(around:{radius}, {lat}, {lon})["office"];
    );
    out center;
    """

    try:
        r = requests.post(overpass_url, data=query.encode("utf-8"), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("elements", [])
    except Exception as e:
        # نرجّع قائمة فاضية مع الرسالة لو حبيت تستخدمها للتشخيص
        return []


def classify_poi(elements):
    """
    نستخرج وصف مختصر لأقرب نشاط من عناصر OSM.
    نرجع:
        - poi_type: نوع النشاط (school, mosque, shop, ... أو other أو no_poi)
        - poi_desc: وصف نصي تجميعي
    """
    if not elements:
        return "no_poi", "لا يوجد أي نشاط قريب في OSM"

    # نأخذ أول عنصر كأقرب نتيجة (أبسط شيء)
    el = elements[0]
    tags = el.get("tags", {})

    amenity = tags.get("amenity")
    building = tags.get("building")
    shop = tags.get("shop")
    office = tags.get("office")
    name = tags.get("name") or tags.get("name:ar") or ""

    # تحديد نوع رئيسي
    poi_type = amenity or shop or building or office or "other"

    parts = []
    if name:
        parts.append(f"الاسم: {name}")
    if amenity:
        parts.append(f"amenity={amenity}")
    if building:
        parts.append(f"building={building}")
    if shop:
        parts.append(f"shop={shop}")
    if office:
        parts.append(f"office={office}")

    poi_desc = " | ".join(parts) if parts else "نشاط غير محدد بدقة"

    return poi_type, poi_desc


def is_site_active(poi_type, poi_desc):
    """
    نقرر هل الموقع "يبدو مستخدم" بناءً على نوع النشاط والوصف.
    هذه قواعد مبدئية، تقدر تعدّلها حسب خبرتك.
    """
    if poi_type == "no_poi":
        return False

    text = (poi_type + " " + poi_desc).lower()

    active_keywords = [
        "school", "university", "college", "kindergarten",
        "hospital", "clinic", "pharmacy",
        "mosque", "place_of_worship",
        "shop", "market", "supermarket", "mall",
        "restaurant", "cafe", "hotel",
        "government", "office", "bank",
        "residential", "commercial", "apartments",
    ]

    if any(k in text for k in active_keywords):
        return True

    # افتراض: أي building عام يعتبر موقع محتمل الاستخدام
    if "building=" in text:
        return True

    return False


def is_freeze_reason_inactive(reason_text):
    """
    نحدد هل سبب التجميد معناه (الموقع غير مستخدم / مزال / مهجور ...).
    نعتمد على كلمات عربية في عمود السبب.
    عدّل قائمة الكلمات حسب سبب التجميد عندكم.
    """
    if not isinstance(reason_text, str):
        return False

    t = reason_text.replace(" ", "").lower()

    keywords = [
        "قابلللسقوط",
        "ازالة", "إزالة",
        "غيرنشط",
        "مفصول",       # عداد مفصول من المصدر
        "مفقود",
        "مزال", "مزالمنالشركة",
        "موقفلعدم",   # موقّف لعدم الاستهلاك
        "مهجور",
    ]

    return any(k.replace(" ", "") in t for k in keywords)


def build_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"


# ----------------------------
# واجهة المستخدم
# ----------------------------

st.sidebar.header("📂 إعدادات الإدخال")

uploaded_file = st.sidebar.file_uploader("حمّل ملف العدادات (Excel/CSV)", type=["xlsx", "xls", "csv"])

radius = st.sidebar.number_input("نصف قطر البحث حول كل عداد (متر)", min_value=10, max_value=200, value=30, step=5)
sample_limit = st.sidebar.number_input("أقصى عدد عدادات لتحليلها (لمنع الضغط على الخادم)", min_value=1, max_value=2000, value=200, step=10)

overpass_url = st.sidebar.text_input("رابط خادم Overpass", value=DEFAULT_OVERPASS_URL)

run_button = st.sidebar.button("تشغيل التحليل")


if uploaded_file is None:
    st.info("⬅️ من فضلك حمّل ملف العدادات من القائمة الجانبية.")
    st.stop()


# قراءة الملف
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"خطأ في قراءة الملف: {e}")
    st.stop()

st.success(f"تم تحميل الملف. عدد الصفوف: {len(df)}")

# اختيار أعمدة الإحداثيات والسبب
st.subheader("🔧 ربط الأعمدة")

lat_col = st.selectbox("اختر عمود خط العرض (latitude)", options=df.columns, index=list(df.columns).index("latitude") if "latitude" in df.columns else 0)
lon_col = st.selectbox("اختر عمود خط الطول (longitude)", options=df.columns, index=list(df.columns).index("longitude") if "longitude" in df.columns else 1)

reason_col = st.selectbox("اختر عمود سبب التجميد (اختياري)", options=["(بدون سبب)"] + list(df.columns))
has_reason = reason_col != "(بدون سبب)"

id_col = st.selectbox("اختر عمود رقم العداد أو المعرّف (اختياري)", options=["(بدون)"] + list(df.columns))


if not run_button:
    st.stop()

# تقليل عدد العينات لو الملف كبير
if len(df) > sample_limit:
    st.warning(f"سيتم تحليل أول {sample_limit} عداد فقط لتقليل الضغط على خادم Overpass.")
    df = df.head(sample_limit)

progress_bar = st.progress(0)
results = []

st.subheader("🚀 جاري تنفيذ الاستعلامات على OSM...")

for idx, row in df.iterrows():
    lat = row[lat_col]
    lon = row[lon_col]

    # تخطي الإحداثيات المفقودة أو صفرية
    try:
        if pd.isna(lat) or pd.isna(lon) or float(lat) == 0 or float(lon) == 0:
            results.append({
                "poi_type": "no_coord",
                "poi_desc": "إحداثيات غير صالحة",
                "site_active": False,
                "status": "no_coord"
            })
            continue
    except Exception:
        results.append({
            "poi_type": "no_coord",
            "poi_desc": "إحداثيات غير صالحة",
            "site_active": False,
            "status": "no_coord"
        })
        continue

    elements = call_overpass(lat, lon, radius, overpass_url=overpass_url)
    poi_type, poi_desc = classify_poi(elements)
    site_active = is_site_active(poi_type, poi_desc)

    # سبب التجميد
    reason_text = row[reason_col] if has_reason else ""
    reason_inactive = is_freeze_reason_inactive(str(reason_text)) if has_reason else False

    # تصنيف الحالة
    if not has_reason:
        status = "no_reason"
    elif reason_inactive and site_active:
        status = "suspicious"  # تجميد غير منطقي
    elif reason_inactive and not site_active:
        status = "freeze_ok"
    else:
        status = "not_clear"

    results.append({
        "poi_type": poi_type,
        "poi_desc": poi_desc,
        "site_active": site_active,
        "reason_inactive": reason_inactive,
        "status": status
    })

    progress_bar.progress((idx + 1) / len(df))
    time.sleep(1)  # مهم عشان ما نزعج خادم Overpass


# دمج النتائج مع البيانات الأصلية
res_df = df.copy().reset_index(drop=True)
res_extra = pd.DataFrame(results)
res_df = pd.concat([res_df, res_extra], axis=1)

# روابط قوقل ماب
res_df["google_maps"] = res_df.apply(lambda r: build_google_maps_link(r[lat_col], r[lon_col]), axis=1)

# جدول للحالات المشبوهة
st.subheader("⚠️ الحالات المشبوهة (تجميد غير منطقي محتمل)")
suspicious = res_df[res_df["status"] == "suspicious"]
st.write(f"عدد الحالات المشبوهة: {len(suspicious)}")
st.dataframe(suspicious)

# عرض جميع النتائج
st.subheader("📊 جميع النتائج")
st.dataframe(res_df)

# تحميل النتائج كملف Excel
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    res_df.to_excel(writer, index=False, sheet_name="all")
    suspicious.to_excel(writer, index=False, sheet_name="suspicious")

st.download_button(
    label="⬇️ تحميل النتائج كملف Excel",
    data=output.getvalue(),
    file_name="frozen_meters_with_osm_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
