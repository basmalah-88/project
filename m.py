import streamlit as st
import pandas as pd
import io

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

# Load data
uploaded_file = st.file_uploader("Upload your updated_file.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

"""
try:
    df = pd.read_csv('C:/Users/user/Downloads/updated_file.csv')
except FileNotFoundError:
    st.error("Error: 'updated_file.csv' not found. Please ensure the file path is correct.")
    st.stop() # Stop execution if file not found
"""
# Set up Streamlit UI
st.title('NABLS-AI')
st.info('NABLS-AI: Trend Analysis in Artificial Intelligence Research')

# Display raw data
st.subheader("Raw Data Preview")
st.dataframe(df)

# User-editable code block
cody = '''
# Ensure 'Publication date' is in datetime format and 'Year' is extracted
df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')
df['Year'] = df['Publication date'].dt.year

# Fill NaN in 'Model' and 'Abstract' with empty strings before concatenating
df['Model'] = df['Model'].fillna('')
df['Abstract'] = df['Abstract'].fillna('')


def classify_trend(model_name):
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
    elif 'Computer-Using Agent' in model_name:
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
    elif 'INTELLECT' in model_name or 'k0' in model_name:
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
    elif 'Movie' in model_name or 'Gen' in model_name:
        return 'AI Generation'
    elif 'Nemotron' in model_name:
        return 'Large-Scale,Open-Source AI'
    elif 'NVLM' in model_name:
        return 'Multimodal AI,Open-Source AI'
    elif 'Octo' in model_name:
        return 'Robotics AI,Open-Source AI'
    elif 'o1' in model_name or 'o3' in model_name:
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
    elif 'Veo' in model_name or 'Wan' in model_name:
        return 'AI Generation'
    elif 'Yi' in model_name:
        return 'Large-Scale,Efficiency'
    else:
        return 'Other'

df['AI_Trend'] = df['Model'].apply(classify_trend)

df['AI_Trend'] = df['AI_Trend'].str.split(',')
df = df.explode('AI_Trend')
df['AI_Trend'] = df['AI_Trend'].str.strip()

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

def classify_ai_trend(text):
    if pd.isna(text):
        return None
    text = text.lower()
    for trend, keywords in trend_keywords.items():
        if any(keyword in text for keyword in keywords):
            return trend
    return "Other"

mask_other = df['AI_Trend'] == "Other"
df.loc[mask_other, 'AI_Trend_Reclassified'] = (
    df.loc[mask_other, ['Abstract', 'Model']]
    .fillna('')
    .agg(' '.join, axis=1)
    .apply(classify_ai_trend)
)

df['AI_Trend'] = df.apply(
    lambda row: row['AI_Trend_Reclassified'] if row['AI_Trend'] == "Other" and row['AI_Trend_Reclassified'] != "Other"
    else row['AI_Trend'],
    axis=1
)

df.drop(columns=['AI_Trend_Reclassified'], inplace=True)

# Calculate the frequency of each AI_Trend
AI_Trends_Frequency_df = df['AI_Trend'].value_counts().reset_index()
AI_Trends_Frequency_df.columns = ['AI_Trend', 'Frequency']
st.subheader("Frequency of each AI Trend:")
st.dataframe(AI_Trends_Frequency_df)

# Calculate trends by year
trend_by_year = df.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')

# Get the top 5 AI_Trends based on total count
top_5_trends = AI_Trends_Frequency_df.nlargest(5, columns='Frequency')['AI_Trend'].tolist()

# Filter the trend_by_year DataFrame to include only the top 5 trends
filtered_top_5 = trend_by_year[trend_by_year['AI_Trend'].isin(top_5_trends)]

# Plot Evolution of Top 5 AI Trends Over Years
st.subheader('Evolution of Top 5 AI Trends Over Years')
fig1, ax1 = plt.subplots(figsize=(14, 7))
sns.lineplot(data=filtered_top_5, x='Year', y='Model_Count', hue='AI_Trend', marker='o', ax=ax1)
ax1.set_title('Evolution of Top 5 AI Trends Over Years')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Models')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Display processed dataframe
st.subheader("Processed Data Preview (with AI_Trend)")
st.dataframe(df)

# Drop rows with missing 'Domain'
df_clean = df.dropna(subset=['Domain'])

# Split multi-domain entries (e.g., "Multimodal,Language") into separate rows
df_exploded = df_clean.assign(Domain=df_clean['Domain'].str.split(',')).explode('Domain')
df_exploded['Domain'] = df_exploded['Domain'].str.strip()

# Count number of models per domain per year
domain_trends = df_exploded.groupby(['Year', 'Domain']).size().unstack(fill_value=0)

st.subheader("Domain Trends Tail")
st.dataframe(domain_trends.tail())

# Plot Trends in the Top 5 Most Popular Domains
st.subheader("Trends in the Top 5 Most Popular Domains (2021–2025)")
top_domains = domain_trends.sum().sort_values(ascending=False).head(5).index
fig2, ax2 = plt.subplots(figsize=(14, 6))
domain_trends[top_domains].plot(marker='o', linewidth=1, ax=ax2)
ax2.set_title("Trends in the Top 5 Most Popular Domains (2021–2025)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of AI Models")
ax2.legend(title="Number of Models")
plt.tight_layout()
st.pyplot(fig2)

# Plot All AI Domains
st.subheader("Trends in All AI Model Domains (2021–2025)")
fig3, ax3 = plt.subplots(figsize=(16, 8))
domain_trends.plot(marker='.', linewidth=1, alpha=0.7, ax=ax3, legend=False)
ax3.set_title("Trends in AI Model Domains (2021–2025)")
ax3.set_xlabel("Year")
ax3.set_ylabel("Number of AI Models")
plt.tight_layout()
st.pyplot(fig3)

# Plot Evolution of All AI Trends Over Years
st.subheader('Evolution of All AI Trends Over Years')
all_trends = df['AI_Trend'].unique()
filtered = trend_by_year[trend_by_year['AI_Trend'].isin(all_trends)]

for trend in all_trends:
    fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
    data = filtered[filtered['AI_Trend'] == trend]
    if not data.empty:
        sns.lineplot(data=data, x='Year', y='Model_Count', marker='o', ax=ax_trend)
        ax_trend.set_title(f'Evolution of {trend} Over Years')
        ax_trend.set_ylabel('Number of Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_trend)
    else:
        st.write(f"No data to plot for {trend} over the years.")

st.subheader("Top 5 AI Trends:")
st.write(top_5_trends)

# Step 2: Prepare Trend Data for prediction
st.subheader("Predicted Model Counts for 2026 for Top 5 Trends")
filtered_trends_2021_2025 = trend_by_year[
    (trend_by_year['AI_Trend'].isin(top_5_trends)) &
    (trend_by_year['Year'] >= 2021) &
    (trend_by_year['Year'] <= 2025)
].copy()

# Step 3: Build and Train Trend Models
trend_models = {}
trend_predictions_2026 = {}

for trend in top_5_trends:
    trend_data = filtered_trends_2021_2025[filtered_trends_2021_2025['AI_Trend'] == trend]

    if not trend_data.empty:
        X = trend_data['Year'].values.reshape(-1, 1)
        y = trend_data['Model_Count'].values

        model = LinearRegression()
        model.fit(X, y)
        trend_models[trend] = model

        prediction_2026 = model.predict([[2026]])
        trend_predictions_2026[trend] = max(0, int(round(prediction_2026[0])))
    else:
        trend_predictions_2026[trend] = 0

st.write("Linear Regression Models Trained for Top 5 Trends.")
for trend, count in trend_predictions_2026.items():
    st.write(f"{trend}: {count}")

# Step 9: Summarize and Plot Predictions
st.subheader('Predicted Number of Models for Top 5 AI Trends in 2026')
predictions_df = pd.DataFrame(list(trend_predictions_2026.items()), columns=['AI_Trend', 'Predicted_Model_Count_2026'])
predictions_df = predictions_df.sort_values(by='Predicted_Model_Count_2026', ascending=False)

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(data=predictions_df, x='Predicted_Model_Count_2026', y='AI_Trend', palette='viridis', ax=ax4)
ax4.set_title('Predicted Number of Models for Top 5 AI Trends in 2026')
ax4.set_xlabel('Predicted Number of Models')
ax4.set_ylabel('AI Trend')
plt.tight_layout()
st.pyplot(fig4)

# Plot Number of models per trend per year
st.subheader('Number of models per trend per year')
df_plot = df.dropna(subset=['Year', 'AI_Trend']).copy()
trend_per_year = df_plot.groupby(['Year', 'AI_Trend']).size().reset_index(name='Model_Count')
pivot_table = trend_per_year.pivot(index='Year', columns='AI_Trend', values='Model_Count').fillna(0)

fig5, ax5 = plt.subplots(figsize=(12, 6))
pivot_table.plot(kind='bar', stacked=True, colormap='tab20', ax=ax5)
ax5.set_title('Number of models per trend per year')
ax5.set_xlabel('Year')
ax5.set_ylabel('Number of models')
ax5.legend(title='AI Trend', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig5)

# Top Model Producers
st.subheader("Top Model Producers")
org_counts = df['Organization'].value_counts().reset_index()
org_counts.columns = ['Organization', 'Model_Count']
st.dataframe(org_counts.head(10))

fig6, ax6 = plt.subplots(figsize=(12, 6))
sns.barplot(data=org_counts.head(10), x='Model_Count', y='Organization', palette='viridis', ax=ax6)
ax6.set_title('Top 10 AI Organization by Number of Models')
ax6.set_xlabel('Number of Models')
ax6.set_ylabel('Organization')
plt.tight_layout()
st.pyplot(fig6)

# Clustering and Classification
st.subheader("Clustering and Classification Analysis")

# Convert 'Frontier model' to string type to handle boolean values
df['Frontier model'] = df['Frontier model'].astype(str)

# Select relevant features for clustering
relevant_features = ['Parameters', 'Training compute (FLOP)', 'Training dataset size (datapoints)',
                     'Epochs', 'Training time (hours)', 'Hardware quantity',
                     'Training compute cost (2023 USD)', 'Batch size',
                     'Organization categorization', 'Country (of organization)',
                     'Model accessibility', 'Training code accessibility',
                     'Inference code accessibility', 'Frontier model', 'AI_Trend', 'Year']

# Separate features into numerical and categorical
numerical_features = [col for col in relevant_features if col in df.columns and df[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in relevant_features if col in df.columns and df[col].dtype == 'object']

# Ensure all relevant features are present in the DataFrame before preprocessing
missing_features = [col for col in relevant_features if col not in df.columns]
if missing_features:
    st.warning(f"Warning: The following relevant features are missing from your data and will be excluded: {', '.join(missing_features)}")
    relevant_features = [col for col in relevant_features if col not in missing_features]
    numerical_features = [col for col in numerical_features if col not in missing_features]
    categorical_features = [col for col in categorical_features if col not in missing_features]


# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply preprocessing to the dataframe
try:
    df_preprocessed = preprocessing_pipeline.fit_transform(df[relevant_features])
    st.write(f"Shape of preprocessed data: {df_preprocessed.shape}")
except KeyError as e:
    st.error(f"Error during preprocessing: A column mentioned in relevant_features was not found. {e}")
    st.stop()


# Apply KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_preprocessed)

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters
st.write("Cluster assignments added to the dataframe.")
st.dataframe(df[['Model', 'AI_Trend', 'Cluster']].head())

# Evaluate and interpret clusters
st.subheader("Numerical Feature Means per Cluster:")
numerical_cluster_summary = df.groupby('Cluster')[numerical_features].mean()
st.dataframe(numerical_cluster_summary)

st.subheader("Categorical Feature Value Counts per Cluster:")
for feature in categorical_features:
    st.write(f"--- {feature} ---")
    st.dataframe(df.groupby('Cluster')[feature].value_counts().unstack(fill_value=0))

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_preprocessed)

# Create a new DataFrame for the PCA results and add the cluster labels
df_pca_plot = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca_plot['Cluster'] = clusters

# Visualize the clusters using a scatter plot
fig7, ax7 = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=df_pca_plot, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', legend='full', ax=ax7)
ax7.set_title('K-Means Clusters Visualized with PCA')
ax7.set_xlabel('PCA Component 1')
ax7.set_ylabel('PCA Component 2')
st.pyplot(fig7)

# Supervised Classification (Predicting Cluster)
st.subheader("Supervised Classification (Predicting Cluster)")
# For supervised learning, use the preprocessed data and the created clusters
X = df_preprocessed
y = df['Cluster'] # Use 'Cluster' from the original df now that it's added

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

st.write(f"Shape of X_train: {X_train.shape}")
st.write(f"Shape of X_test: {X_test.shape}")
st.write(f"Shape of y_train: {y_train.shape}")
st.write(f"Shape of y_test: {y_test.shape}")

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
st.write("Supervised classification model trained successfully.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

st.write(f"Model Accuracy: {accuracy:.4f}")
st.write(f"Model Precision (weighted): {precision:.4f}")
st.write(f"Model Recall (weighted): {recall:.4f}")

st.subheader("Predicting Cluster for a New Model")
new_model_data = pd.DataFrame({
    'Parameters': [1.0e11],
    'Training compute (FLOP)': [5.0e24],
    'Training dataset size (datapoints)': [1.0e12],
    'Epochs': [100],
    'Training time (hours)': [500],
    'Hardware quantity': [1000],
    'Training compute cost (2023 USD)': [100000],
    'Batch size': [1000000],
    'Organization categorization': ['Industry'],
    'Country (of organization)': ['United States of America'],
    'Model accessibility': ['Open weights (unrestricted)'],
    'Training code accessibility': ['Open source'],
    'Inference code accessibility': ['Open source'],
    'Frontier model': ['True'], # Changed to string as per preprocessing
    'AI_Trend': ['Multimodal AI'],
    'Year': [2025]
})

# Preprocess the new data point
new_model_preprocessed = preprocessing_pipeline.transform(new_model_data[relevant_features])

# Predict the cluster
predicted_cluster = model.predict(new_model_preprocessed)
st.write(f"Predicted Cluster for the new model: {predicted_cluster[0]}")

# Infer dominant AI trends in the predicted cluster
dominant_trends_in_cluster = df[df['Cluster'] == predicted_cluster[0]]['AI_Trend'].value_counts().head(3)
st.write(f"Dominant AI Trends in Predicted Cluster {predicted_cluster[0]}:")
st.dataframe(dominant_trends_in_cluster)

'''

user_code = st.text_area("✍️ عدلي الكود هنا:", cody, height=800)

if st.button("run"):
    if user_code:
        try:
            # Clear previous plots and outputs to avoid duplication when re-running
            st.empty()
            # Execute the user's code
            exec(user_code, globals(), locals())
        except Exception as e:
            st.error(f"حدث خطأ أثناء تنفيذ الكود: {e}")
    else:
        st.warning("يرجى إدخال كود Python للتنفيذ.")
