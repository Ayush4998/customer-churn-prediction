# Customer Churn Prediction using XGBoost
# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import xgboost as xgb
import pickle

# Load and prepare data
# Replace with your actual data loading paths
df = pd.concat([
    pd.read_csv('../data/customer_churn_dataset-training-master.csv'),
    pd.read_csv('../data/customer_churn_dataset-testing-master.csv')
], axis=0)
df.reset_index(drop=True, inplace=True)

# Initial preprocessing
df.drop(columns='CustomerID', inplace=True)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
df.dropna(inplace=True)

# Convert discrete columns to int
discrete_col = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'last_interaction']
for col in discrete_col:
    df[col] = df[col].astype(int)

# Train-test split
y = df['churn']
X = df.drop(columns='churn')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Reset indices
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# One Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[['gender', 'subscription_type', 'contract_length']])
feature_names = encoder.get_feature_names_out(['gender', 'subscription_type', 'contract_length'])

# Transform train and test data
train_categorical_one_encoded_data = encoder.transform(X_train[['gender', 'subscription_type', 'contract_length']])
train_OHE_df = pd.DataFrame(train_categorical_one_encoded_data, columns=feature_names)

test_categorical_one_encoded_data = encoder.transform(X_test[['gender', 'subscription_type', 'contract_length']])
test_OHE_df = pd.DataFrame(test_categorical_one_encoded_data, columns=feature_names)

# Remove original categorical columns
X_train = X_train.drop(columns=['gender', 'subscription_type', 'contract_length'])
X_test = X_test.drop(columns=['gender', 'subscription_type', 'contract_length'])

# Concatenate encoded features
X_train = pd.concat([X_train, train_OHE_df], axis=1)
X_test = pd.concat([X_test, test_OHE_df], axis=1)

# Train XGBoost model
xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Evaluate model
y_pred = xgb_classifier.predict(X_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print()

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print()

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Save the XGBoost model
with open("customer_churn_xgboost_model.pkl", 'wb') as model_file:
    pickle.dump(xgb_classifier, model_file)

# Save the encoder
with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("XGBoost model and encoder saved successfully!")

# Deployment class for predictions
class CustomerChurnClassifier:
    def __init__(self, model_path, encoder_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        with open(encoder_path, 'rb') as file:
            self.encoder = pickle.load(file)
    
    def predict(self, age: int, tenure: int, usage_frequency: int, support_calls: int, 
                payment_delay: int, total_spend: float, last_interaction: int, 
                gender: str, subscription_type: str, contract_length: str):
        
        # Checking input datatypes
        expected_data_types = [int, int, int, int, int, float, int, str, str, str]
        input_arguments = [age, tenure, usage_frequency, support_calls, payment_delay, 
                          total_spend, last_interaction, gender, subscription_type, contract_length]
        input_arguments_names = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 
                                'total_spend', 'last_interaction', 'gender', 'subscription_type', 'contract_length']
        
        for i in range(len(input_arguments)):
            current_arg_type = type(input_arguments[i])
            if current_arg_type != expected_data_types[i]:
                raise TypeError(f"Error: Given {input_arguments_names[i]} ({current_arg_type}) is not of expected type {expected_data_types[i]}")
        
        # Checking gender, subscription_type, and contract_length values
        valid_genders = ['Female', 'Male']
        valid_subscription_types = ['Standard', 'Basic', 'Premium']
        valid_contract_lengths = ['Annual', 'Monthly', 'Quarterly']
        
        if gender not in valid_genders:
            raise ValueError(f"Error: Invalid gender value '{gender}'. Expected one of {valid_genders}")
        if subscription_type not in valid_subscription_types:
            raise ValueError(f"Error: Invalid subscription_type value '{subscription_type}'. Expected one of {valid_subscription_types}")
        if contract_length not in valid_contract_lengths:
            raise ValueError(f"Error: Invalid contract_length value '{contract_length}'. Expected one of {valid_contract_lengths}")
        
        # One Hot Encoding
        ohe_data = list(self.encoder.transform([[gender, subscription_type, contract_length]])[0])
        to_predict_array = [age, tenure, usage_frequency, support_calls, payment_delay, 
                           total_spend, last_interaction] + ohe_data
        to_predict_array = np.array(to_predict_array).reshape((1, -1))
        
        prediction = self.model.predict(to_predict_array)[0]
        
        if prediction > 0.5:
            return 'Will Churn'
        else:
            return "Won't Churn"

# Example usage:
# customer_churn = CustomerChurnClassifier(
#     model_path='customer_churn_xgboost_model.pkl',
#     encoder_path='encoder.pkl'
# )
# 
# result = customer_churn.predict(
#     age=19,
#     tenure=48,
#     usage_frequency=7,
#     support_calls=3,
#     payment_delay=30,
#     total_spend=787.0,
#     last_interaction=29,
#     gender='Female',
#     subscription_type='Premium',
#     contract_length='Annual'
# )
# print(result)
