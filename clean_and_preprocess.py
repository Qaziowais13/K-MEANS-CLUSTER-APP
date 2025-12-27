import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'dataset final.csv'
df = pd.read_csv(file_path)

# Drop the target variable if present
if 'Target' in df.columns:
    df = df.drop('Target', axis=1)

# Handle missing values (simple strategy: drop rows with any missing values)
df_clean = df.dropna()

# Encode categorical variables if any (all columns are numeric except possibly 'Course')
if df_clean['Course'].dtype == 'object':
    df_clean['Course'] = df_clean['Course'].astype('category').cat.codes

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Save cleaned and scaled data for next steps
np.save('X_scaled.npy', X_scaled)
print('Data cleaned and preprocessed. Shape:', X_scaled.shape)