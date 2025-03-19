# save_model.py
import torch

def save_model(model, scaler, file_path):
    torch.save({"model_state_dict": model.state_dict(), "scaler": scaler, "input_size": model.lstm1.input_size}, file_path)
    print(f"Model saved to {file_path}")