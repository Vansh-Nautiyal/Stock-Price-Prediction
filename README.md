# Stock Price Prediction using LSTM



This project implements a **Long Short-Term Memory (LSTM)** neural network to predict future stock prices based on historical data.  

The model is intentionally kept simple and beginner-friendly, using only the **closing price** for training and prediction.



### This project is for educational purposes only and should not be used for real financial trading.



---



## 🚀  Project Features



- 📊 Fetches real-time historical stock data using \*\*Yahoo Finance\*\*

- 🧠 LSTM-based deep learning model

- 🔁 Recursive multi-day future prediction

- 📉 Visualizes actual vs predicted prices

- 🖥️ Interactive \*\*Streamlit\*\* web interface

- 🧪 Beginner-friendly ML project structure



---



## 🧠 Model Architecture



- Input: Previous **60 days** of closing prices

- Layers:

  - LSTM (64 units, return sequences)
  - Dropout (0.2)

  - LSTM (32 units)
  - Dropout (0.2)

  - Dense (1 output)

- Optimizer: **Adam**

- Loss Function: **Mean Squared Error (MSE)**



---



## 📊 Dataset



- Source: `yfinance`

- Feature used: **Close price only**

- Date range: User selectable

- Stock symbol: User selectable



---



## 🖥️ How to Run the Project

```bash
1. Clone the repository

git clone https://github.com/YOUR\_USERNAME/stock-price-prediction.git
cd stock-price-prediction



2. Create and activate virtual environment (for Windows)

python -m venv venv
venv\\Scripts\\activate      



3. Install Dependencies

pip install -r requirements.txt



4. Run the Streamlit app

streamlit run app.py





Project Structure

Stock-Price-Prediction/

│

├── data/

│   └── raw/

│       ├── AAPL.csv          # Apple stock historical data

│       ├── INFY.csv          # Infosys stock data

│       ├── MSFT.csv          # Microsoft stock data

│       └── TSLA.csv          # Tesla stock data

│

├── models/

│   └── (saved models, ignored in GitHub)

│

├── src/

│   ├── \_\_pycache\_\_/          # Python cache files

│   ├── data\_loader.py       # Loads stock data

│   ├── preprocess.py        # Data cleaning and scaling

│   ├── feature\_engineering.py  # Sequence creation for LSTM

│   ├── model\_lstm.py        # LSTM model architecture

│   ├── evaluate.py          # Model evaluation

│   └── main.py              # Training and prediction pipeline

│

├── app.py                   # Streamlit application entry point

├── Actual\_vs\_Predicted.png  # Sample prediction visualization

├── .gitignore               # Files ignored by Git

├── requirements.txt         # Project dependencies

└── README.md                # Project documentation



