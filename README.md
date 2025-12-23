# SMS Spam / Ham Classification

This repository contains an **SMS Spam Detection application** built using **Deep Learning (RNN)** and deployed with **Streamlit**. The model classifies incoming SMS messages as **Spam** or **Ham (Not Spam)**.

🌐 **Live Streamlit App:**  
https://sms-spam-ham.streamlit.app/

---

## Project Overview

- SMS text classification using Deep Learning
- RNN-based model trained with TensorFlow/Keras
- Deployed as an interactive Streamlit web app
- Uses a pre-trained model for fast predictions

---

## Features

- Classifies SMS messages as Spam or Ham
- Simple and clean Streamlit UI
- Real-time predictions
- Pre-trained model hosted on GitHub

---

## Dataset

The model is trained on the **SMS Spam Collection Dataset**, which is publicly available and widely used for spam detection tasks.

Source:
https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection

The dataset contains labeled SMS messages:
- `spam`
- `ham`

---

## Project Structure

SMS-Spam-Ham/  
├── app.py              # Streamlit application  
├── smsspam.py          # Model training script  
├── spam_mod.h5         # Trained model file  
├── requirements.txt    # Dependencies  
└── README.md  

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Setup Instructions (Local)

1. Clone the repository  
git clone https://github.com/Sohan-DS/SMS-Spam-Ham.git  
cd SMS-Spam-Ham  

2. (Optional) Create and activate a virtual environment  
python -m venv venv  
source venv/bin/activate  
Windows: venv\Scripts\activate  

3. Install dependencies  
pip install -r requirements.txt  

4. Run the Streamlit app  
streamlit run app.py  

---

## How It Works

- User enters an SMS message
- Text is preprocessed and tokenized
- The trained RNN model predicts the label
- Result is displayed as Spam or Ham

---

## Model Details

- Embedding Layer
- Simple RNN Layer
- Fully Connected Dense Layers
- Sigmoid activation for binary classification

The trained model is saved as `spam_mod.h5` and loaded dynamically in the Streamlit app.

---

## Notes

- Internet connection is required to download the model file
- Prediction accuracy depends on the quality of input text
- This project is intended for learning and demonstration purposes

---

## Future Improvements

- Improve accuracy with LSTM/GRU
- Add probability score to predictions
- Support batch SMS classification
- Deploy with Docker or cloud services

---

## Author

Sohan-DS
