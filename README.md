# 💡 Customer Churn Prediction using Artificial Neural Network (ANN)

This project is an interactive **Streamlit web application** that predicts whether a customer will **leave a bank (churn)** or **stay**, using a trained **Artificial Neural Network (ANN)** built with TensorFlow and Keras.

It is designed to demonstrate how deep learning models can be deployed as a simple web app for real-time predictions.

---

## 🧠 What This Project Does

The goal of this project is to:
- Predict customer churn (Yes/No) based on various features.
- Allow users to input data via a **Streamlit interface**.
- Use a pre-trained **ANN model (`model.h5`)** to make predictions.
- Load previously trained **encoders** and **scalers** (`.pkl` files) for preprocessing user inputs.

This project is a complete machine learning pipeline — from preprocessing to model deployment.

---

## ⚙️ How the Code Works

### 🧩 1. Data Preprocessing
Before training, the data is cleaned and encoded:
- **Categorical features** (like Geography, Gender) are transformed using **OneHotEncoder**.
- **Numerical features** are standardized using **StandardScaler**.
- These encoders and the scaler are saved using **Pickle** so they can be reused during prediction:
  ```python
  with open('scaler.pkl', 'rb') as file:
      scaler = pickle.load(file)
