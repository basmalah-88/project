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
st.title('NABLS-AI: Trends in Artificial Intelligence Research')
st.info('NABLS-AI:A tool for analyzing data and exploring trends in artificial intelligence research.')

# --- Download data from a fixed path (within the repository) ---
st.header("Source data 📊")

# === Important: Make sure that the ‘ai_models.csv’ file is located in the same folder as your Streamlit application file ===
# Or if it is inside a subfolder, for example'data/ai_models.csv'
file_path = 'ai_models.csv' # If it is in the same folder as the application
# file_path = 'data/ai_models.csv' # If there is a subfolder named 'data'

try:
    df = pd.read_csv(file_path)
    st.success(f"The file was downloaded from: `{file_path}`")
    st.subheader("Preview raw data")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error(f"Error: File ‘{file_path}’ does not exist. Please check the correct path and the existence of the file in your project folder on GitHub.")
    st.stop() # Suspend execution if file not found

#--Editable code editor ---
st.header("Code modification and data analysis ✍️")
st.markdown("""
    You can modify the code below to perform your own analyses.
    **Important notes:**
    * **Do not reload `df` from other CSV files within the code unless absolutely necessary.** The variable `df` already contains the loaded data.
    * **Use `st.write()` for text, `st.dataframe()` for tables, and `st.pyplot()` for graphs.**
* **All attempts to save files to fixed local paths (`df.to_csv`) have been removed from this code because they will not work in a web environment.**
""")

# Default code for the user to edit
cody = '''
# Start of analysis: By Noora, Sedrah, Basmaleh

# Cleaning and processing data - (by Noora)
# df is the DataFrame that was already loaded at the beginning of the application.
# Do not reload the data here unless you need another file.

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

    st.write(f"The unique values of Confidence: {df['Confidence'].unique()}")
    if columns_to_drop:
        df.drop(columns=columns_to_drop, axis=1, inplace=True)
        st.write(f"The columns have been knocked down: {columns_to_drop}")
    else:
        st.write("No columns were dropped based on the minimum threshold")

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
else:
    st.warning("The ‘Confidence’ column does not exist. Processing of this column has been skipped.")

 #**Note: `df.to_csv` has been omitted here to prevent path errors in a web environment.


# K-Nearest Neighbors (KNN) Analysis - (by Noora)
# df is the DataFrame that has already been loaded and processed.
# Do not reload the data here.

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
        st.warning("There is too little training data to build an error rate chart forـ KNN.")
    else:
        for i in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))

        fig_knn_error, ax_knn_error = plt.subplots(figsize=(10,6))
        ax_knn_error.plot(range(1, max_k + 1), error_rate, color='blue', linestyle='dashed', marker='o',
                          markerfacecolor='red', markersize=10)
        ax_knn_error.set_title('Error rate versus value K')
        ax_knn_error.set_xlabel('value of K')
        ax_knn_error.set_ylabel('Error rate')
        ax_knn_error.grid(True)
        st.pyplot(fig_knn_error)
        plt.close(fig_knn_error)
else:
    st.warning("The ‘Confidence’ column does not exist, analysis cannot be performed KNN.")


# Clean and prepare data (by Sedrah)
# df is the DataFrame that has already been loaded and processed.
# Do not reload the data here.

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
    st.write(f"The unique values of AI_Trend after initial classification: {df['AI_Trend'].unique()}")
else:
    st.warning("column 'Model' Not available. Artificial intelligence trends will not be classified.")
    df['AI_Trend'] = 'Other'

if 'Publication date' in df.columns:
    df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')
    df['Year'] = df['Publication date'].dt.year
else:
    st.warning("The ‘Publication date’ column is missing. The year will not be calculated.")
    df['Year'] = 0

# **Note: `df.to_csv` has been omitted here to prevent path errors in a web environment.**

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
    st.warning("Columns required for classifying artificial intelligence trends (AI_Trend, Abstract, Model) are missing.")

# **Note: `df.to_csv` has been omitted here to prevent path errors in a web environment.**


# Analysis of annual trends (by Sedrah)
df = df[df['AI_Trend'] != 'Other']
if not df.empty and 'Year' in df.columns and 'AI_Trend' in df.columns:
    trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')
    st.subheader("Artificial intelligence trends by year")
    st.dataframe(trend_by_year.head())
else:
    st.warning("DataFrame is empty or columns ‘Year’/'AI_Trend' are missing after filtering. Annual trends cannot be calculated.")
    trend_by_year = pd.DataFrame(columns=['Year', 'AI_Trend', 'Model_Count'])


# Visualizing Trends (by Noora)
if 'AI_Trend' in df.columns and not df.empty and not trend_by_year.empty:
    AI_Trends_Frequency_df= df['AI_Trend'].value_counts().reset_index()
    AI_Trends_Frequency_df.columns = ['AI_Trend', 'Frequency']
    st.subheader("Repeating each direction of artificial intelligence")
    st.dataframe(AI_Trends_Frequency_df)

    top_5_trends = AI_Trends_Frequency_df.nlargest(5, columns='Frequency')['AI_Trend'].tolist()
    filtered_top_5 = trend_by_year[trend_by_year['AI_Trend'].isin(top_5_trends)]

    if not filtered_top_5.empty:
        sns.set(style="whitegrid")
        fig_top5, ax_top5 = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=filtered_top_5, x='Year', y='Model_Count', hue='AI_Trend', marker='o', ax=ax_top5)
        ax_top5.set_title('The evolution of the top 5 trends in artificial intelligence over the years')
        ax_top5.set_xlabel('Year')
        ax_top5.set_ylabel('Number of models')
        ax_top5.tick_params(axis='x', rotation=45)
        ax_top5.legend(title='Artificial Intelligence Trends')
        plt.tight_layout()
        st.pyplot(fig_top5)
        plt.close(fig_top5)
    else:
        st.warning("No data to plot the top 5 trends in artificial intelligence after filtering.")
else:
    st.warning("Cannot calculate AI trend frequency, column ‘AI_Trend’ is missing or DataFrame is empty.")


# Field Trend Analysis (by Noora)
st.subheader("Field Trend Analysis")
if 'Domain' in df.columns and not df.empty:
    df_clean = df.dropna(subset=['Domain'])
    df_exploded = df_clean.assign(Domain=df_clean['Domain'].str.split(',')).explode('Domain')
    df_exploded['Domain'] = df_exploded['Domain'].str.strip()

    if 'Year' in df_exploded.columns:
        domain_trends = df_exploded.groupby(['Year', 'Domain']).size().unstack(fill_value=0)
        st.write("Field directions (tail):")
        st.dataframe(domain_trends.tail())
    else:
        st.warning("The ‘Year’ column is missing in the DataFrame field. Field trends cannot be calculated by year.")
        domain_trends = pd.DataFrame()
else:
    st.warning("The ‘Domain’ column is missing or the DataFrame is empty. Domain trend analysis cannot be performed.")
    domain_trends = pd.DataFrame()


st.warning("The ‘Domain’ column is missing or the DataFrame is empty. Domain trends cannot be analyzed.") # Visualize domain trends (by Noora)
st.subheader("Domain trend graphs")
if not domain_trends.empty:
    sns.set(style="whitegrid")
    fig_top_domains, ax_top_domains = plt.subplots(figsize=(14, 6))
    top_domains = domain_trends.sum().sort_values(ascending=False).head(5).index
    if not top_domains.empty:
        domain_trends[top_domains].plot(marker='o', linewidth=1, ax=ax_top_domains)
        ax_top_domains.set_title("Top 5 Popular Domains (2021–2025)")
        ax_top_domains.set_xlabel("Year")
        ax_top_domains.set_ylabel("Number of AI Models")
        ax_top_domains.legend(title="Number of Models"))
        plt.tight_layout()
        st.pyplot(fig_top_domains)
        plt.close(fig_top_domains)
    else:
        st.warning("No upper domains found for the graph.")

    fig_all_domains, ax_all_domains = plt.subplots(figsize=(16, 8))
    domain_trends.plot(marker='.', linewidth=1, alpha=0.7, ax=ax_all_domains, legend=False)
    ax_all_domains.set_title("Trends in AI Model Domains (2021–2025)")
    ax_all_domains.set_xlabel("Year")
    ax_all_domains.set_ylabel("Number of AI Models")
    plt.tight_layout()
    st.pyplot(fig_all_domains)
    plt.close(fig_all_domains)
else:
    st.warning("No data for domain trends to visualize.")


# Evolution of individual AI trends (by Sedrah)
st.subheader("Evolution of individual AI trends")
if 'AI_Trend' in df.columns and not df.empty and 'trend_by_year' in locals() and not trend_by_year.empty:
    all_trends = df['AI_Trend'].unique()
    filtered = trend_by_year[trend_by_year['AI_Trend'].isin(all_trends)]

    for trend in all_trends:
        fig_single_trend, ax_single_trend = plt.subplots(figsize=(10, 5))
        data = filtered[filtered['AI_Trend'] == trend]
        if not data.empty:
            sns.lineplot(data=data, x='Year', y='Model_Count', marker='o', ax=ax_single_trend)
            ax_single_trend.set_title(f'Evolution of {trend} over the years')
            ax_single_trend.set_ylabel(trend)
            ax_single_trend.tick_params(axis=‘x’, rotation=45)
            plt.tight_layout()
            st.pyplot(fig_single_trend)
            plt.close(fig_single_trend)
        else:
            st.write(f"No data for trend chart: {trend}")
else:
    st.warning("Individual trend evolution cannot be displayed. Insufficient data or missing columns.")


# Regression model for predicting AI trends in 2026 (by Noora)
st.subheader("Regression model for predicting AI trends in 2026")

if ‘top_5_trends’ in locals() and top_5_trends and ‘trend_by_year’ in locals() and not trend_by_year.empty:
    st.write(f"Top 5 AI trends identified for prediction: {top_5_trends}")

    filtered_trends_2021_2025 = trend_by_year[
        (trend_by_year['AI_Trend'].isin(top_5_trends)) &
        (trend_by_year['Year'] >= 2021) &
        (trend_by_year['Year'] <= 2025)
    ].copy()

    st.write("Filtered trend data (2021-2025) for the top 5 trends:")
    st.dataframe(filtered_trends_2021_2025)

    trend_models = {}
    trend_predictions_2026 = {}

    for trend in top_5_trends:
        trend_data = filtered_trends_2021_2025[filtered_trends_2021_2025['AI_Trend'] == trend]

        if not trend_data.empty and len(trend_data) >= 2:
            X = trend_data['Year'].values.reshape(-1, 1)
            y = trend_data['Model_Count'].values
            # st.write(f"the data for {trend} (X): {X}")
            # st.write(f"the data for {trend} (y): {y}")

            model = LinearRegression()
            model.fit(X, y)
            trend_models[trend] = model
            prediction_2026 = model.predict([[2026]])
            trend_predictions_2026[trend] = max(0, int(round(prediction_2026[0])))
        else:
            st.write(f"Insufficient data for trend: {trend} in the years 2021-2025 to train the model. You may need more points.")
            trend_predictions_2026[trend] = 0

    st.write("Linear regression models for the top 5 trends have been trained.")
    st.write("Number of models predicted for 2026:")
    for trend, count in trend_predictions_2026.items():
        st.write(f"{trend}: {count}")

    predictions_df = pd.DataFrame(list(trend_predictions_2026.items()), columns=['AI_Trend', 'Predicted_Model_Count_2026'])
    predictions_df = predictions_df.sort_values(by='Predicted_Model_Count_2026', ascending=False)

    sns.set(style="whitegrid")
    fig_pred_bar, ax_pred_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(data=predictions_df, x='Predicted_Model_Count_2026', y='AI_Trend', palette='viridis', ax=ax_pred_bar)
    ax_pred_bar.set_title('Number of models predicted for the top 5 AI trends in 2026')
    ax_pred_bar.set_xlabel('Number of models predicted')
    ax_pred_bar.set_ylabel('AI trend')
    plt.tight_layout()
    st.pyplot(fig_pred_bar)
    plt.close(fig_pred_bar)
else:
    st.warning("Regression analysis cannot be performed. The top 5 trends for artificial intelligence have not been identified, are empty, or annual trend data is not available.")


# Top AI organizations by number of models (by Basmaleh)
st.subheader("Top AI organizations by number of models")
if 'Organization' in df.columns and not df.empty:
    org_counts = df['Organization'].value_counts().reset_index()
    org_counts.columns = ['Organization', 'Model_Count']

    st.write("Best model producers:")
    st.dataframe(org_counts.head(10))

    fig_org_top10, ax_org_top10 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=org_counts.head(10), x='Model_Count', y='Organization', palette='viridis', ax=ax_org_top10)
    ax_org_top10.set_title('Top 10 AI organizations by number of models')
    ax_org_top10.set_xlabel('Number of models')
    ax_org_top10.set_ylabel('Organization')
    plt.tight_layout()
    st.pyplot(fig_org_top10)
    plt.close(fig_org_top10)
else:
    st.warning("Cannot display top organizations. ‘Organization’ column is missing or DataFrame is empty.")
'''

# Use code_editor for a better editing experience (optional)
try:
    from code_editor import code_editor
    response_dict = code_editor(cody, lang="python", height=800,
                                editor_props={"theme": "dracula"})
    user_code_from_editor = response_dict['text'] if response_dict and 'text' in response_dict else cody
except ImportError:
    st.warning("`streamlit-code-editor` is not installed. Please install it (pip install streamlit-code-editor) for a better editor experience, or `st.text_area` will be used.")
    user_code_from_editor = st.text_area("✍️ Edit the code here:", cody, height=800)


# --- Run button ---
if st.button("Run analysis"):
    if not user_code_from_editor.strip():
        final_user_code = cody
    else:
        final_user_code = user_code_from_editor

    if 'df' in locals() and df is not None: 
        try:
            st.empty() 

            # Create a copy of the DataFrame to avoid modifying the original df with user code.
            df_for_exec = df.copy()

            # Specify the environment in which the user code will be executed
            exec_globals = {
                'st': st, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
                'df': df_for_exec, # Pass the modified version
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

            st.success("The code was successfully executed!")
            st.write("---")
            st.write("The analysis is complete. You can modify the code above and restart.")

        except Exception as e:
            st.error(f"An error occurred while executing the code: {e}")
            st.exception(e) # To display the complete error trace
    else:
       st.warning("Please make sure to download the data file first.")
