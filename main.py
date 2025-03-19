# main.py
from datetime import datetime
import os
import sys
import torch
from get_and_save_data import get_and_save_data
from load_and_prepare_data import load_and_prepare_data
from load_or_create_model import load_or_create_model
from plot_prediction import plot_prediction
from predict_future import predict_future
from save_model import save_model
from train_model import train_model

def main(file_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_steps = 12

    full_df, training_set, prediction_set = load_and_prepare_data(file_path, time_steps, "2023-12-31")
    model = load_or_create_model(model_path, device, training_set)

    train_model(model, training_set, device, 0.2, 32, 50, 1)
    train_model(model, training_set, device, 0.2, 32, 50, 2) # Resample training/testing data before 2023 for the next 50 epochs to predict more accurately
    save_model(model, training_set.scaler(), model_path)

    predictions = predict_future(model, prediction_set, device)
    plot_prediction(full_df, predictions)

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        api_key_file = "fred_api_key"
        try:
            with open(api_key_file, "r") as f:
                fred_api_key = f.read().strip()  # Remove any trailing newlines or spaces
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file '{api_key_file}' not found. Please create it with your FRED API key.")
        
        file_path = f"fed_rate_data_{datetime.now().strftime('%Y%m%d')}.csv"
        if not os.path.exists(file_path):
            get_and_save_data(fred_api_key, file_path)
    elif len(args) == 2:
        file_path = args[1]
    else:
        raise "Too many arguments."
    
    main(file_path, "model.pt")