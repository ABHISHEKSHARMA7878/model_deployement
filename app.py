import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv('Titanic_train.csv')

# Preprocessing steps
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Convert categorical columns to dummy variables
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Prepare features and target variable
X_train = train_data.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y_train = train_data['Survived']

# Scale numerical features
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Train the model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Save the model and scaler
with open('log_reg_model.pkl', 'wb') as model_file:
    pickle.dump(log_reg, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

st.title("Titanic Survival Prediction")

st.write("""
This app predicts whether a passenger survived the Titanic disaster using a logistic regression model.
""")

# Function to preprocess user input
def preprocess_input(data):
    data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
    missing_cols = set(X_train.columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    data = data[X_train.columns]
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    return data

# User input section
st.sidebar.header('User Input Features')
def user_input_features():
    Pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    Age = st.sidebar.slider('Age', 0, 80, 29)
    SibSp = st.sidebar.slider('SibSp', 0, 8, 0)
    Parch = st.sidebar.slider('Parch', 0, 6, 0)
    Fare = st.sidebar.slider('Fare', 0, 513, 32)
    Embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    data = {'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': Embarked}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess input data
processed_input = preprocess_input(input_df)

# Predict using the loaded model
prediction = log_reg.predict(processed_input)
prediction_prob = log_reg.predict_proba(processed_input)

# Display results
st.subheader('Prediction')
st.write('Survived' if prediction[0] == 1 else 'Did Not Survive')

st.subheader('Prediction Probability')
st.write(f"Survived: {prediction_prob[0][1]:.2f}, Did Not Survive: {prediction_prob[0][0]:.2f}")

# Additional Visualizations
st.subheader('Feature Coefficients')
coefficients = pd.DataFrame(log_reg.coef_.T, index=processed_input.columns, columns=['Coefficient'])
st.write(coefficients.sort_values(by='Coefficient', ascending=False))

st.subheader('Confusion Matrix')
conf_matrix = confusion_matrix(y_train, log_reg.predict(X_train))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='coolwarm')
plt.title('Confusion Matrix')
st.pyplot(plt)

# ROC Curve
st.subheader('ROC Curve')
fpr, tpr, _ = roc_curve(y_train, log_reg.predict_proba(X_train)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
st.pyplot(plt)