import streamlit as st
import datetime
from src.main import run_prediction

st.title("Stock Price Predictor")
st.subheader("A time-series stock prediction project that analyzes historical market data, engineers technical indicators, and uses an LSTM neural network to forecast future closing prices.")

st.markdown("---")
options = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Motors",
    "INFY": "Infosys Ltd."
}

choice = st.selectbox(
    "Select Stock Symbol",
    [f"{k} - {v}" for k, v in options.items()],
    index=None,
    placeholder="Choose a stock symbol..."
)

if choice:
    symbol = choice.split(" - ")[0]   # Extract actual ticker


start = st.date_input(
    "Start Date",
    min_value=datetime.date(2000, 1, 1),
    max_value=datetime.date.today()
)

end = st.date_input(
    "End Date",
    min_value=datetime.date(2000, 1, 1),
    max_value=datetime.date.today()
)

epochs = st.number_input("Select Epochs : ",min_value=0,max_value=50,step=5)

predict=st.button("Predict price")
if predict:
    if choice==None:
        st.warning("Choose a stock")
    elif epochs==0:
        st.warning("Number of Epochs cannot be zero !")
    else:
        status = st.empty()
        result=run_prediction(symbol,start,end,status,epochs)
        st.subheader("Actual v/s prediction Plot")
        st.pyplot(result["prediction_plot"])
        st.subheader("Loss Plot")
        st.pyplot(result["loss_plot"])

        st.subheader("Future Seven day prices: ")
        st.pyplot(result["future_prices"])
