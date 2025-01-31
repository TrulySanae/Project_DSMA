# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = '../2) Data Cleaning/6b) Imputed_Dataset.csv'
data = pd.read_csv(file_path)
data = data.drop('business_id', axis=1, errors='ignore')

# Separate features and target variable
y = data['rating']                 # Target variable
X = data.drop(columns=['rating'])  # Features


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
# Initialize SMOTE
smote = SMOTE(random_state=42)
# Resample the training set
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Combine resampled features and target into a DataFrame
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['rating'])], axis=1)

# Save the resampled dataset to a new CSV file
resampled_data.to_csv('1b) Resampled_Dataset.csv', index=False)

print("Resampled dataset saved to '1b) Resampled_Dataset.csv'.")
