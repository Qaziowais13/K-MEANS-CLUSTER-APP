
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="K-means Clustering App", layout="centered")

# Email input screen
if 'email_valid' not in st.session_state:
    st.session_state['email_valid'] = False

def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

if not st.session_state['email_valid']:
    st.title("🔒 Enter Your Email to Continue")
    email = st.text_input("Email Address", key="email_input")
    if st.button("Continue"):
        if is_valid_email(email):
            st.session_state['email_valid'] = True
            st.session_state['user_email'] = email
            st.success("Email accepted! You can now use the clustering app.")
            st.rerun()
        else:
            st.error("Please enter a valid email address.")
    st.stop()

# Main clustering app
st.title("🔬 Modern K-means Clustering GUI")
st.write("Upload your dataset (CSV), select number of clusters, and visualize the results interactively.")

dataset = st.file_uploader("Upload CSV Dataset", type=["csv"])

if dataset is not None:
    df = pd.read_csv(dataset)
    st.write("### Data Preview", df.head())
    target_col = st.selectbox("Select target column to drop (if any):", [None] + list(df.columns))
    if target_col:
        df = df.drop(target_col, axis=1)
    if df.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Rows with missing values will be dropped.")
        df = df.dropna()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype('category').cat.codes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    st.subheader("Elbow Method for Optimal k")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, marker='o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    st.success(f"K-means Inertia: {kmeans.inertia_:.2f}")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig2, ax2 = plt.subplots()
    for cluster in range(k):
        ax2.scatter(X_pca[labels == cluster, 0], X_pca[labels == cluster, 1], label=f'Cluster {cluster+1}')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('K-means Clusters (PCA)')
    ax2.legend()
    st.pyplot(fig2)
    df_result = df.copy()
    df_result['Cluster'] = labels
    st.download_button("Download Clustered Data as CSV", df_result.to_csv(index=False), "clustered_data.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Made with Streamlit · Modern K-means Clustering · GPT-4.1")
