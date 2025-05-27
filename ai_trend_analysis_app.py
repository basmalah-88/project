import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

st.title("📊 تحليل اتجاهات الذكاء الاصطناعي")

uploaded_file = st.file_uploader("📁 ارفعي ملف CSV الخاص بالنماذج", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ تم تحميل الملف بنجاح")

    st.subheader("👁️‍🗨️ معاينة أولية للبيانات")
    st.write(df.head())

    df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')
    df['Year'] = df['Publication date'].dt.year

    st.subheader("📈 عدد الأوراق لكل سنة")
    papers_per_year = df['Year'].value_counts().sort_index()
    st.bar_chart(papers_per_year)

    st.subheader("📌 التصنيفات الأكثر شيوعًا")
    if 'Category' in df.columns:
        st.bar_chart(df['Category'].value_counts())

    st.subheader("📉 تحليل الانحدار لعدد الأوراق بمرور السنوات")
    df_yearly = df.groupby('Year').size().reset_index(name='Count').dropna()
    X = df_yearly[['Year']]
    y = df_yearly['Count']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label='البيانات الحقيقية')
    ax.plot(X, y_pred, color='red', label='خط الانحدار')
    ax.set_xlabel("السنة")
    ax.set_ylabel("عدد الأوراق")
    ax.set_title("📉 الانحدار الخطي")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔍 التحليل العنقودي باستخدام K-Means")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_features:
        X_cluster = df[numeric_features].dropna()

if not X_cluster.empty:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    ax2.set_title("نتائج التجميع")
    st.pyplot(fig2)
else:
    st.warning("⚠️ لا توجد بيانات عددية كافية بدون قيم مفقودة لتحليل التجميع.")


    st.subheader("🧠 نموذج تصنيف باستخدام Random Forest")
    df_model = df.dropna()
    target_column = st.selectbox("🎯 اختاري العمود المستهدف للتصنيف", df_model.columns)
    feature_columns = st.multiselect("🧩 اختاري الأعمدة المستخدمة كميزات", df_model.columns.drop(target_column))

    if feature_columns:
        X = df_model[feature_columns]
        y = df_model[target_column]

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier())])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("🔎 دقة النموذج:", accuracy_score(y_test, y_pred))
        st.write("🎯 الاسترجاع:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        st.write("🎯 الدقة:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
