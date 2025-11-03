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
🧠 2. Model Training (Offline)

An Artificial Neural Network (ANN) was trained using TensorFlow/Keras:

Input Layer → Hidden Layers → Output Layer (Sigmoid activation)

Binary classification: churn or not churn

Model saved as model.h5 after training:
```python
model.save('model.h5')


💻 3. Streamlit App (Online Prediction)

In app.py:

The trained model and preprocessing objects are loaded.

Streamlit provides input fields for user data.

User inputs are encoded and scaled.

The model predicts churn probability and displays the result in real time.

Example snippet:
```python
model = tf.keras.models.load_model('model.h5')
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

📁 Project Structure
📦 churn-prediction-app
 ┣ 📄 app.py                   # Streamlit application
 ┣ 📄 model.h5                 # Trained ANN model
 ┣ 📄 onehot_encoder_geo.pkl   # Encoder for Geography
 ┣ 📄 onehot_encoder_gender.pkl # Encoder for Gender (if used)
 ┣ 📄 scaler.pkl               # StandardScaler for numeric columns
 ┣ 📄 requirements.txt         # Python dependencies
 ┗ 📄 README.md                # Project documentation

🚀 How to Run the Project
Install Dependencies

Make sure you have Python 3.8+ installed.

🖥️ On your computer (VS Code / Command Prompt):
pip install -r requirements.txt

☁️ On Google Colab:
```python
!pip install -r requirements.txt

# Run the Streamlit App

Start the app with this command:
```python
streamlit run app.py


You’ll see output like:
```python
Local URL: http://localhost:8501
Network URL: http://your-ip:8501


Click the Local URL to open the web app in your browser.

🧮 How to Use the App

Enter customer information in the sidebar (credit score, geography, gender, age, balance, etc.).

Click “Predict”.

The app will:

Preprocess the inputs (encode + scale)

Run them through the trained ANN model

Display a prediction:

✅ Customer will stay

⚠️ Customer likely to churn

Example Output:
```python 
Prediction: Customer likely to stay
Probability: 0.82 (82% confidence)
