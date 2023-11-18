# LSTM-SPY-HF-PREDICTION
High frequency prediction of the SPY etf, using non-proprietary data from lobster.
## How to use?
Get the data from https://lobsterdata.com/.  It's too big to upload to github.
Order Book Data Prediction Model
This project focuses on developing a predictive model using order book data. The goal is to measure and analyze the reactions to significant market events, such as earnings calls, using historical order book data.

Description
The model aims to predict the post-earnings announcement drift (PEAD) for stocks in the S&P 500. Key periods for measurement include:

From the announcement to the next market close.
The end of the next week (5 business days).
Approximately one month later (21 business days).
Up to the subsequent quarterly earnings date.
The current version of the model successfully collects earnings dates, surprises, analyst estimates, and actual values, as well as the time until the market opens. Challenges remain in accurately calculating the PEAD, primarily due to data collection issues.

Installation and Usage
The code is written in Python and requires basic libraries such as NumPy, Pandas, Matplotlib, Seaborn, and TensorFlow. Additionally, Google Colab is used for data manipulation and model training. To run the code:

Mount your Google Drive to access datasets:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Load the order book data:

python
Copy code
orderbook_data = pd.read_csv('PATH/TO/YOUR/DATA.csv')
Process and normalize the data, create data generators, and split the data into training, validation, and testing sets.

Build, compile, and train the LSTM model.

Evaluate the model's performance using metrics like MAE, MSE, and RMSE.

Model Architecture
The model utilizes Long Short-Term Memory (LSTM) networks, a type of recurrent neural network, to capture temporal dependencies in order book data. Key components include:

Data preprocessing and normalization.
Sequential data generation for training and testing.
LSTM layers with dropout and regularization for learning complex patterns.
Dense output layer for prediction.
Limitations and Challenges
Data Quality: Issues with data collection can significantly impact the model's accuracy.
PEAD Calculation: Challenges remain in the precise calculation of post-earnings announcement drift.
Future Work
Improving data collection methods to ensure data quality.
Refining the PEAD calculation methodology.
Experimenting with different model architectures and hyperparameters.
Contributions
Contributions to the project are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

