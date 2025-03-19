# predict_future.py
import numpy as np
import torch

def predict_future(model, prediction_set, device):
    model.eval()
    predictions = []
    input_data = prediction_set.x
    
    with torch.no_grad():
        for i in range(len(prediction_set)):
            x_tensor = input_data[i].unsqueeze(0).to(device) # x_tensor.shape = (1, 12, 10)
            pred = model(x_tensor)
            pred_value = pred.cpu().numpy()[0, 0]
            predictions.append(pred_value)

    predictions = prediction_set.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()