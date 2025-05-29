import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Set up Streamlit UI
st.set_page_config(layout="wide", page_title="NABLS-AI Dashboard")
st.title('NABLS-AI: تحليل الاتجاهات في أبحاث الذكاء الاصطناعي')
st.info('NABLS-AI: أداة لتحليل البيانات واستكشاف الاتجاهات في أبحاث الذكاء الاصطناعي.')

# --- تحميل البيانات من مسار ثابت ---
st.header("بيانات المصدر 📊")

# === هام: الرجاء تغيير هذا المسار إلى المسار الفعلي لملفك ===
# استخدم المسار بنظام الشرطة المائلة الأمامية / أو أضف 'r' قبل المسار
file_path = 'C:/Users/BAB AL SAFA/Desktop/nablsai/ai_models.csv'
# مثال على مسار نسبي إذا كان الملف في نفس مجلد تطبيق Streamlit:
# file_path = 'ai_models.csv'

try:
    df = pd.read_csv(file_path)
    st.success(f"تم تحميل الملف من: {file_path}")
    st.subheader("معاينة البيانات الأولية")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error(f"خطأ: الملف '{file_path}' غير موجود. الرجاء التأكد من المسار الصحيح.")
    st.stop() # إيقاف التنفيذ إذا لم يتم العثور على الملف

# --- محرر الكود القابل للتعديل ---
st.header("تعديل الكود وتحليل البيانات ✍️")
st.markdown("""
    يمكنك تعديل الكود في الأسفل لتنفيذ تحليلاتك الخاصة.
    **ملاحظات هامة:**
    * **لا تقم بإعادة تحميل `df` من ملفات CSV أخرى داخل الكود ما لم يكن ذلك ضروريًا للغاية.** المتغير `df` يحتوي بالفعل على البيانات المحملة.
    * **استخدم `st.write()` للنصوص، `st.dataframe()` للجداول، و `st.pyplot()` للرسوم البيانية.**
    * **تم تعديل جميع مسارات `df.to_csv` داخل الكود إلى مسارات خام (باستخدام `r''`) لضمان التوافق.**
""")

# Default code for the user to edit
cody = '''
# بداية التحليل: By Noora, Sedrah, Basmaleh

# تنظيف البيانات ومعالجتها - (بواسطة Noora)
# df هو DataFrame الذي تم تحميله بالفعل في بداية التطبيق.
# لا تقم بإعادة تحميل البيانات هنا إلا إذا كنت تحتاج إلى ملف آخر.
# df = pd.read_csv(r'C:/Users/BAB AL SAFA/Desktop/nablsai/ai_models.csv')

if 'Confidence' in df.columns:
    df = df.dropna(subset=["Confidence"])
    df['Confidence'] = df['Confidence'].str.strip().str.lower()
    df = df[df['Confidence'] != 'unknown']

    confidence_count = df["Confidence"].count()
    equ = confidence_count * 0.85
    columns_to_drop = []
    for column_name in df.columns:
        current_column_non_null_count = df[column_name].count()
        if current_column_non_null_count < equ:
            columns_to_drop.append(column_name)

    st.write(f"القيم الفريدة لـ Confidence: {df['Confidence'].unique()}")
    if columns_to_drop:
        df.drop(columns=columns_to_drop, axis=1, inplace=True)
        st.write(f"تم إسقاط الأعمدة: {columns_to_drop}")
    else:
        st.write("لم يتم إسقاط أي أعمدة بناءً على الحد الأدنى.")

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
else:
    st.warning("عمود 'Confidence' غير موجود. تم تخطي معالجة هذا العمود.")

# حفظ التغييرات إلى ملف جديد (اختياري، وقد لا يكون فعالاً في بيئات النشر السحابية)
df.to_csv(r'C:/Users/BAB AL SAFA/Desktop/nablsai/updated_file3.csv', index=False)

# تحليل K-Nearest Neighbors (KNN) - (بواسطة Noora)
# df هو DataFrame الذي تم تحميله ومعالجته بالفعل.
# لا تقم بإعادة تحميل البيانات هنا.
# df = pd.read_csv(r'C:/Users/BAB AL SAFA/Desktop/nablsai/updated_file3.csv')

if 'Confidence' in df.columns:
    X = df.drop("Confidence", axis=1)
    y = df["Confidence"]

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)

    X = pd.get_dummies(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    st.subheader("نتائج نموذج K-Nearest Neighbors (KNN)")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    st.write("Confusion Matrix:")
    st.code(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    error_rate = []
    max_k = min(40, len(X_train) // 2)
    if max_k < 1:
        st.warning("بيانات التدريب قليلة جدًا لبناء مخطط معدل الخطأ لـ KNN.")
    else:
        for i in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))

        fig_knn_error, ax_knn_error = plt.subplots(figsize=(10,6))
        ax_knn_error.plot(range(1, max_k + 1), error_rate, color='blue', linestyle='dashed', marker='o',
                        markerfacecolor='red', markersize=10)
        ax_knn_error.set_title('معدل الخطأ مقابل قيمة K')
        ax_knn_error.set_xlabel('قيمة K')
        ax_knn_error.set_ylabel('معدل الخطأ')
        ax_knn_error.grid(True)
        st.pyplot(fig_knn_error)
        plt.close(fig_knn_error)
else:
    st.warning("عمود 'Confidence' غير موجود، لا يمكن إجراء تحليل KNN.")


# تنظيف البيانات وإعدادها (بواسطة Sedrah)
# df هو DataFrame الذي تم تحميله ومعالجته بالفعل.
# لا تقم بإعادة تحميل البيانات هنا.
# df = pd.read_csv(r'C:/Users/BAB AL SAFA/Desktop/nablsai/ai_models.csv')

def classify_trend(model_name):
    if pd.isna(model_name): return 'Other'
    model_name = str(model_name)
    if 'AFM-on-device' in model_name:
        return 'On-Device AI,Closed Source,Efficiency'
    elif 'AFM-server' in model_name:
        return 'Large-Scale,Closed Source,AI Regulation'
    elif 'AlphaProteo' in model_name:
        return 'AI for Science,Closed Source'
    elif 'Amazon Nova Pro' in model_name:
        return 'Multimodal AI,Closed Source,Large-Scale'
    elif 'Cambrian' in model_name:
        return 'Multimodal AI,Open-Source AI,Domain-Specific'
    elif 'CHAI' in model_name:
        return 'AI for Science,Open-Source AI'
    elif 'Claude' in model_name:
        return 'Reasoning-Mathematics,Closed Source,AI Regulation'
    elif 'Computer-Using Agent ' in model_name:
        return 'Robotics AI,Closed Source,Multimodal AI'
    elif 'DeepL' in model_name:
        return 'Domain-Specific,Closed Source'
    elif 'DeepSeek-R1' in model_name:
        return 'Reasoning-Mathematics,Open-Source AI'
    elif 'DeepSeek-V2.5' in model_name:
        return 'Domain-Specific,Open-Source AI'
    elif 'DeepSeek-V3' in model_name:
        return 'Large-Scale,Open-Source AI'
    elif 'Doubao' in model_name:
        return 'Large-Scale,Closed Source'
    elif 'ERNIE' in model_name:
        return 'Multimodal AI,Closed Source'
    elif 'ESM' in model_name:
        return 'AI for Science,Open-Source AI'
    elif 'EXAONE' in model_name:
        return 'Reasoning-Mathematics,Open-Source AI'
    elif 'Fugatto' in model_name:
        return 'AI Generation,Open-Source AI'
    elif 'Gemin' in model_name:
        return 'Multimodal AI,Closed Source'
    elif 'GLM' in model_name:
        return 'Large-Scale,Closed Source'
    elif 'GPT-4.5' in model_name:
        return 'Large-Scale,Multimodal AI,Closed Source'
    elif 'GPT-4o' in model_name:
        return 'Multimodal AI,Closed Source,Efficiency'
    elif 'GR' in model_name:
        return 'Robotics AI,Open-Source AI'
    elif 'Hairuo' in model_name:
        return 'Domain-Specific'
    elif 'Hunyuan' in model_name:
        return 'Large-Scale,Open-Source AI,Multimodal AI'
    elif 'Infinity' in model_name:
        return 'AI Generation,Open-Source AI'
    elif ('INTELLECT' in model_name) or ('k0' in model_name):
        return 'Reasoning-Mathematics,Closed Source'
    elif 'LLaVA-OV-72B' in model_name:
        return 'Multimodal AI,Open-Source AI'
    elif 'Llama 3' in model_name:
        return 'Large-Scale,Open-Source AI,Multimodal AI'
    elif 'Llama 4' in model_name:
        return 'Ultra-Large-Scale,Multimodal AI,Closed Source'
    elif 'Mathstral' in model_name:
        return 'Reasoning-Mathematics,Open-Source AI'
    elif 'Mercury' in model_name:
        return 'Efficiency,Closed Source'
    elif 'Mistral' in model_name:
        return 'Large-Scale,Domain-Specific'
    elif ('Movie' in model_name) or ('Gen' in model_name):
        return 'AI Generation'
    elif 'Nemotron' in model_name:
        return 'Large-Scale,Open-Source AI'
    elif 'NVLM' in model_name:
        return 'Multimodal AI,Open-Source AI'
    elif 'Octo' in model_name:
        return 'Robotics AI,Open-Source AI'
    elif ('o1' in model_name) or ('o3' in model_name):
        return 'Reasoning-Mathematics,Efficiency'
    elif 'Open' in model_name:
        return 'Robotics AI,Open-Source AI'
    elif 'Pangu' in model_name:
        return 'Large-Scale'
    elif 'Palmyra' in model_name:
        return 'Domain-Specific'
    elif 'Pixtral' in model_name:
        return 'Multimodal AI,Open-Source AI'
    elif 'Qwen' in model_name:
        return 'Large-Scale,Open-Source AI'
    elif 'QwQ' in model_name:
        return 'Reasoning-Mathematics,Open-Source AI'
    elif 'Table' in model_name:
        return 'Robotics AI'
    elif ('Veo' in model_name) or ('Wan' in model_name):
        return 'AI Generation'
    elif 'Yi' in model_name:
        return 'Large-Scale,Efficiency'
    else:
        return 'Other'

if 'Model' in df.columns:
    df['AI_Trend'] = df['Model'].apply(classify_trend)
    df['AI_Trend'] = df['AI_Trend'].str.split(',')
    df = df.explode('AI_Trend')
    df['AI_Trend'] = df['AI_Trend'].str.strip()
    st.write(f"القيم الفريدة لـ AI_Trend بعد التصنيف الأولي: {df['AI_Trend'].unique()}")
else:
    st.warning("عمود 'Model' غير موجود. لن يتم تصنيف اتجاهات الذكاء الاصطناعي.")
    df['AI_Trend'] = 'Other'

if 'Publication date' in df.columns:
    df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')
    df['Year'] = df['Publication date'].dt.year
else:
    st.warning("عمود 'Publication date' غير موجود. لن يتم حساب السنة.")
    df['Year'] = 0

# حفظ التغييرات إلى ملف جديد (اختياري)
df.to_csv(r'C:/Users/BAB AL SAFA/Desktop/nablsai/updated_file.csv', index=False)

trend_keywords = {
    'AI Generation': ['generate', 'generative', 'GAN', 'diffusion', 'image-to-image', 'synthesis'],
    'Efficiency': ['efficient', 'efficiency', 'dropout', 'compress', 'optimize', 'low-resource', 'reduce'],
    'Multimodal AI': ['multimodal', 'vision-language', 'text and image', 'audio and text'],
    'Robotics AI': ['robot', 'embodied', 'navigation', 'control', 'manipulation'],
    'Reasoning-Mathematics': ['math', 'reasoning', 'proof', 'solve equation'],
    'AI for Science': ['biology', 'chemistry', 'protein', 'scientific', 'molecule'],
    'Closed Source': ['not released', 'private', 'not available'],
    'Open-Source AI': ['open-source', 'released', 'public', 'github'],
    'Large-Scale': ['large dataset', 'scale', 'GPT', 'BERT', 'Transformer'],
    'Ultra-Large-Scale': ['100B', 'trillion', 'huge model', 'massive'],
    'On-Device AI': ['on-device', 'edge', 'mobile', 'low-power'],
    'AI Regulation': ['policy', 'ethics', 'governance', 'regulation', 'safety'],
    'Domain-Specific': ['medical', 'legal', 'finance', 'healthcare', 'education'],
}

def classify_ai_trend_reclassify(text):
    if pd.isna(text):
        return "Other"
    text = str(text).lower()
    for trend, keywords in trend_keywords.items():
        if any(keyword in text for keyword in keywords):
            return trend
    return "Other"

if 'AI_Trend' in df.columns and 'Abstract' in df.columns and 'Model' in df.columns:
    mask_other = df['AI_Trend'] == "Other"
    df.loc[mask_other, 'AI_Trend_Reclassified'] = (
        df.loc[mask_other, ['Abstract', 'Model']]
        .fillna('')
        .agg(' '.join, axis=1)
        .apply(classify_ai_trend_reclassify)
    )

    df['AI_Trend'] = df.apply(
        lambda row: row['AI_Trend_Reclassified'] if row['AI_Trend'] == "Other" and row['AI_Trend_Reclassified'] != "Other"
        else row['AI_Trend'],
        axis=1
    )
    df.drop(columns=['AI_Trend_Reclassified'], inplace=True, errors='ignore')
else:
    st.warning("أعمدة مطلوبة لتصنيف اتجاهات الذكاء الاصطناعي (AI_Trend, Abstract, Model) مفقودة.")

# حفظ التغييرات إلى ملف جديد (اختياري)
df.to_csv(r'C:/Users/BAB AL SAFA/Desktop/nablsai/updated_file.csv', index=False)


# تحليل الاتجاهات السنوية (بواسطة Sedrah)
df = df[df['AI_Trend'] != 'Other']
if not df.empty and 'Year' in df.columns and 'AI_Trend' in df.columns:
    trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')
    st.subheader("اتجاهات الذكاء الاصطناعي حسب السنة")
    st.dataframe(trend_by_year.head())
else:
    st.warning("DataFrame فارغ أو أعمدة 'Year'/'AI_Trend' مفقودة بعد التصفية. لا يمكن حساب الاتجاهات السنوية.")
    trend_by_year = pd.DataFrame(columns=['Year', 'AI_Trend', 'Model_Count'])


# تصور الاتجاهات (بواسطة Noora)
if 'AI_Trend' in df.columns and not df.empty and not trend_by_year.empty:
    AI_Trends_Frequency_df= df['AI_Trend'].value_counts().reset_index()
    AI_Trends_Frequency_df.columns = ['AI_Trend', 'Frequency']
    st.subheader("تكرار كل اتجاه من اتجاهات الذكاء الاصطناعي")
    st.dataframe(AI_Trends_Frequency_df)

    top_5_trends = AI_Trends_Frequency_df.nlargest(5, columns='Frequency')['AI_Trend'].tolist()
    filtered_top_5 = trend_by_year[trend_by_year['AI_Trend'].isin(top_5_trends)]

    if not filtered_top_5.empty:
        sns.set(style="whitegrid")
        fig_top5, ax_top5 = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=filtered_top_5, x='Year', y='Model_Count', hue='AI_Trend', marker='o', ax=ax_top5)
        ax_top5.set_title('تطور أهم 5 اتجاهات للذكاء الاصطناعي عبر السنوات')
        ax_top5.set_xlabel('السنة')
        ax_top5.set_ylabel('عدد النماذج')
        ax_top5.tick_params(axis='x', rotation=45)
        ax_top5.legend(title='اتجاه الذكاء الاصطناعي')
        plt.tight_layout()
        st.pyplot(fig_top5)
        plt.close(fig_top5)
    else:
        st.warning("لا توجد بيانات لرسم أهم 5 اتجاهات للذكاء الاصطناعي بعد التصفية.")
else:
    st.warning("لا يمكن حساب تكرار اتجاهات الذكاء الاصطناعي، العمود 'AI_Trend' مفقود أو DataFrame فارغ.")


# تحليل اتجاهات المجال (بواسطة Noora)
st.subheader("تحليل اتجاهات المجال")
if 'Domain' in df.columns and not df.empty:
    df_clean = df.dropna(subset=['Domain'])
    df_exploded = df_clean.assign(Domain=df_clean['Domain'].str.split(',')).explode('Domain')
    df_exploded['Domain'] = df_exploded['Domain'].str.strip()

    if 'Year' in df_exploded.columns:
        domain_trends = df_exploded.groupby(['Year', 'Domain']).size().unstack(fill_value=0)
        st.write("اتجاهات المجال (الذيل):")
        st.dataframe(domain_trends.tail())
    else:
        st.warning("عمود 'Year' مفقود في DataFrame المجال. لا يمكن حساب اتجاهات المجال حسب السنة.")
        domain_trends = pd.DataFrame()
else:
    st.warning("عمود 'Domain' مفقود أو DataFrame فارغ. لا يمكن إجراء تحليل اتجاهات المجال.")
    domain_trends = pd.DataFrame()


# تصور اتجاهات المجال (بواسطة Noora)
st.subheader("رسوم بيانية لاتجاهات المجال")
if not domain_trends.empty:
    sns.set(style="whitegrid")
    fig_top_domains, ax_top_domains = plt.subplots(figsize=(14, 6))
    top_domains = domain_trends.sum().sort_values(ascending=False).head(5).index
    if not top_domains.empty:
        domain_trends[top_domains].plot(marker='o', linewidth=1, ax=ax_top_domains)
        ax_top_domains.set_title("اتجاهات أهم 5 مجالات شعبية (2021–2025)")
        ax_top_domains.set_xlabel("السنة")
        ax_top_domains.set_ylabel("عدد نماذج الذكاء الاصطناعي")
        ax_top_domains.legend(title="عدد النماذج")
        plt.tight_layout()
        st.pyplot(fig_top_domains)
        plt.close(fig_top_domains)
    else:
        st.warning("لم يتم العثور على مجالات عليا للرسم البياني.")

    fig_all_domains, ax_all_domains = plt.subplots(figsize=(16, 8))
    domain_trends.plot(marker='.', linewidth=1, alpha=0.7, ax=ax_all_domains, legend=False)
    ax_all_domains.set_title("اتجاهات مجالات نماذج الذكاء الاصطناعي (2021–2025)")
    ax_all_domains.set_xlabel("السنة")
    ax_all_domains.set_ylabel("عدد نماذج الذكاء الاصطناعي")
    plt.tight_layout()
    st.pyplot(fig_all_domains)
    plt.close(fig_all_domains)
else:
    st.warning("لا توجد بيانات لاتجاهات المجال لتصورها.")


# تطور اتجاهات الذكاء الاصطناعي الفردية (بواسطة Sedrah)
st.subheader("تطور اتجاهات الذكاء الاصطناعي الفردية")
if 'AI_Trend' in df.columns and not df.empty and 'trend_by_year' in locals() and not trend_by_year.empty:
    all_trends = df['AI_Trend'].unique()
    filtered = trend_by_year[trend_by_year['AI_Trend'].isin(all_trends)]

    for trend in all_trends:
        fig_single_trend, ax_single_trend = plt.subplots(figsize=(10, 5))
        data = filtered[filtered['AI_Trend'] == trend]
        if not data.empty:
            sns.lineplot(data=data, x='Year', y='Model_Count', marker='o', ax=ax_single_trend)
            ax_single_trend.set_title(f'تطور {trend} عبر السنوات')
            ax_single_trend.set_ylabel(trend)
            ax_single_trend.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_single_trend)
            plt.close(fig_single_trend)
        else:
            st.write(f"لا توجد بيانات للرسم البياني للاتجاه: {trend}")
else:
    st.warning("لا يمكن عرض تطور الاتجاهات الفردية. بيانات غير كافية أو أعمدة مفقودة.")


# نموذج الانحدار للتنبؤ باتجاهات الذكاء الاصطناعي في عام 2026 (بواسطة Noora)
st.subheader("نموذج الانحدار للتنبؤ باتجاهات الذكاء الاصطناعي في عام 2026")

if 'top_5_trends' in locals() and top_5_trends and 'trend_by_year' in locals() and not trend_by_year.empty:
    st.write(f"أهم 5 اتجاهات للذكاء الاصطناعي تم تحديدها للتنبؤ: {top_5_trends}")

    filtered_trends_2021_2025 = trend_by_year[
        (trend_by_year['AI_Trend'].isin(top_5_trends)) &
        (trend_by_year['Year'] >= 2021) &
        (trend_by_year['Year'] <= 2025)
    ].copy()

    st.write("بيانات الاتجاهات المصفاة (2021-2025) لأهم 5 اتجاهات:")
    st.dataframe(filtered_trends_2021_2025)

    trend_models = {}
    trend_predictions_2026 = {}

    for trend in top_5_trends:
        trend_data = filtered_trends_2021_2025[filtered_trends_2021_2025['AI_Trend'] == trend]

        if not trend_data.empty and len(trend_data) >= 2:
            X = trend_data['Year'].values.reshape(-1, 1)
            y = trend_data['Model_Count'].values
            # st.write(f"البيانات لـ {trend} (X): {X}")
            # st.write(f"البيانات لـ {trend} (y): {y}")

            model = LinearRegression()
            model.fit(X, y)
            trend_models[trend] = model
            prediction_2026 = model.predict([[2026]])
            trend_predictions_2026[trend] = max(0, int(round(prediction_2026[0])))
        else:
            st.write(f"بيانات غير كافية للاتجاه: {trend} في الأعوام 2021-2025 لتدريب النموذج. قد تحتاج إلى المزيد من النقاط.")
            trend_predictions_2026[trend] = 0

    st.write("تم تدريب نماذج الانحدار الخطي لأهم 5 اتجاهات.")
    st.write("عدد النماذج المتوقعة لعام 2026:")
    for trend, count in trend_predictions_2026.items():
        st.write(f"{trend}: {count}")

    predictions_df = pd.DataFrame(list(trend_predictions_2026.items()), columns=['AI_Trend', 'Predicted_Model_Count_2026'])
    predictions_df = predictions_df.sort_values(by='Predicted_Model_Count_2026', ascending=False)

    sns.set(style="whitegrid")
    fig_pred_bar, ax_pred_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(data=predictions_df, x='Predicted_Model_Count_2026', y='AI_Trend', palette='viridis', ax=ax_pred_bar)
    ax_pred_bar.set_title('عدد النماذج المتوقع لأهم 5 اتجاهات للذكاء الاصطناعي في عام 2026')
    ax_pred_bar.set_xlabel('عدد النماذج المتوقع')
    ax_pred_bar.set_ylabel('اتجاه الذكاء الاصطناعي')
    plt.tight_layout()
    st.pyplot(fig_pred_bar)
    plt.close(fig_pred_bar)
else:
    st.warning("لا يمكن إجراء تحليل الانحدار. لم يتم تحديد أهم 5 اتجاهات للذكاء الاصطناعي أو أنها فارغة، أو بيانات الاتجاهات السنوية غير متاحة.")


# أهم منظمات الذكاء الاصطناعي حسب عدد النماذج (بواسطة Basmaleh)
st.subheader("أهم منظمات الذكاء الاصطناعي حسب عدد النماذج")
if 'Organization' in df.columns and not df.empty:
    org_counts = df['Organization'].value_counts().reset_index()
    org_counts.columns = ['Organization', 'Model_Count']

    st.write("أفضل منتجي النماذج:")
    st.dataframe(org_counts.head(10))

    fig_org_top10, ax_org_top10 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=org_counts.head(10), x='Model_Count', y='Organization', palette='viridis', ax=ax_org_top10)
    ax_org_top10.set_title('أهم 10 منظمات ذكاء اصطناعي حسب عدد النماذج')
    ax_org_top10.set_xlabel('عدد النماذج')
    ax_org_top10.set_ylabel('المنظمة')
    plt.tight_layout()
    st.pyplot(fig_org_top10)
    plt.close(fig_org_top10)
else:
    st.warning("لا يمكن عرض أهم المنظمات. عمود 'Organization' مفقود أو DataFrame فارغ.")
'''

# استخدم code_editor لتجربة تحرير أفضل (اختياري)
try:
    from code_editor import code_editor
    # يمكنك ضبط القيمة الأولية لـ user_code هنا
    response_dict = code_editor(cody, lang="python", height=800,
                                editor_props={"theme": "dracula"}) # يمكنك تغيير المظهر
    # تأكد من أن user_code لا يصبح None إذا لم يكن هناك استجابة من code_editor
    user_code_from_editor = response_dict['text'] if response_dict and 'text' in response_dict else cody
except ImportError:
    st.warning("`streamlit-code-editor` غير مثبت. الرجاء تثبيته (pip install streamlit-code-editor) للحصول على تجربة محرر أفضل، أو سيتم استخدام `st.text_area`.")
    user_code_from_editor = st.text_area("✍️ عدّل الكود هنا:", cody, height=800)


# --- زر التشغيل ---
if st.button("تشغيل التحليل"):
    # إذا لم يقم المستخدم بتغيير الكود، تأكد من استخدام الكود الافتراضي
    # هذا يحل مشكلة "يرجى إدخال كود Python للتنفيذ" عند التشغيل الأول
    # لأن user_code_from_editor قد يكون فارغًا في البداية
    if not user_code_from_editor.strip(): # التحقق مما إذا كانت السلسلة فارغة بعد إزالة المسافات البيضاء
        final_user_code = cody # استخدم الكود الافتراضي
    else:
        final_user_code = user_code_from_editor # استخدم الكود الذي أدخله المستخدم

    if df is not None: # تأكد من تحميل df
        try:
            st.empty() # مسح المخرجات السابقة

            # إنشاء نسخة من DataFrame لتجنب تعديل df الأصلي بواسطة كود المستخدم
            df_for_exec = df.copy()

            # تحديد البيئة التي سيتم فيها تنفيذ كود المستخدم
            exec_globals = {
                'st': st, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
                'df': df_for_exec, # تمرير النسخة المعدلة
                'StandardScaler': StandardScaler, 'OneHotEncoder': OneHotEncoder, 'LabelEncoder': LabelEncoder,
                'ColumnTransformer': ColumnTransformer, 'Pipeline': Pipeline,
                'SimpleImputer': SimpleImputer, 'KMeans': KMeans, 'PCA': PCA,
                'train_test_split': train_test_split, 'RandomForestClassifier': RandomForestClassifier,
                'accuracy_score': accuracy_score, 'precision_score': precision_score,
                'recall_score': recall_score, 'f1_score': f1_score, 'confusion_matrix': confusion_matrix,
                'classification_report': classification_report,
                'LinearRegression': LinearRegression, 'KNeighborsClassifier': KNeighborsClassifier
            }
            exec(final_user_code, exec_globals) # تنفيذ الكود

            st.success("تم تنفيذ الكود بنجاح!")
            st.write("---")
            st.write("انتهى التحليل. يمكنك تعديل الكود أعلاه وإعادة التشغيل.")

        except Exception as e:
            st.error(f"حدث خطأ أثناء تنفيذ الكود: {e}")
            st.exception(e) # لعرض تتبع الخطأ الكامل
    else:
        st.warning("الرجاء التأكد من تحميل ملف البيانات أولاً.")