# FED Rate Prediction

This project aims to predict the Federal Reserve (Fed) interest rate using historical data sourced from the Federal Reserve Economic Data (FRED). By leveraging time series analysis and machine learning techniques, the goal is to model and forecast future Fed rate trends based on past patterns.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Federal Reserve's interest rate decisions have a significant impact on the economy, influencing borrowing costs, investment decisions, and inflation. This project explores historical Fed rate data to build predictive models, providing insights into potential future rate changes.

## Dataset
The data is sourced from [FRED](https://fred.stlouisfed.org/), a comprehensive database maintained by the Federal Reserve Bank of St. Louis. The primary dataset used in this project includes historical Fed interest rates, though additional economic indicators (e.g., inflation, unemployment) could be incorporated for enhanced predictions.

- **Source**: FRED API or downloadable CSV files
- **Time Period**: Varies based on available data (e.g., 1986–present)
- **Target Variable**: Federal Funds Rate
- **Key Variables**: 
    - Personal Consumption Expenditures (Core and Overall)
    - Unemployment Rate
    - Inflation Adjusted GDP
    - M2 Money Supply
    - 2-Year Treasury Yield
    - 10-Year Treasury Yield
    - WTI Crude Oil Price
    - Consumer Sentiment
    - BoJ Immediate Rate

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/a910230/FED_rate_prediction.git
   cd FED_rate_prediction
2. **Install dependencies**:
   Ensure you have Python 3.x installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt

3. **Set up FRED API (optional)**:
   If using the FRED API, obtain an API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) and save it in `fred_api_key` in the project folder (no extension.)

## Usage
1. **Download data automatically, train the model, and make prediction**:
   - `python main.py`

2. **Use csv data to train the model and make prediction**:
   - Example: `python main.py fed_rate_data_20250319.csv`

3. **Visualize results**:
   - Comparison between prediction and historical data as well as the training loss plot is autogenerated as png file.

## Methodology
The project employs time series analysis and machine learning techniques, such as:
- **Data Preprocessing**: Cleaning, handling missing values, and normalizing the dataset.
- **Feature Engineering**: Incorporating lag variables, moving averages, or macroeconomic indicators.
- **Model**: LSTM
- **Evaluation**: RMSE or MAE to assess prediction accuracy.

## Results
Results will include:
- Historical Fed rate trends plotted alongside predicted values.
- Model performance metrics.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes. Suggestions for additional features or datasets are also appreciated.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details."