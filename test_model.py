import pandas as pd
import torch
from anfis_train import SineAndDwellDataset
from torch.utils.data import DataLoader, TensorDataset
from anfis_pytorch import plot_all_mfs, make_anfis
import time

if __name__ == "__main__":
    # set variables
    num_mfs = 3

    SAVE_WEIGHTS_PATH = "data/models/mx5_vars6/state_dict_model.pt"
    # load data
    df = pd.read_csv("data/datasets/mx5_normal_conditions/dataset.csv")
    dataset = SineAndDwellDataset(df)
    train_data = DataLoader(TensorDataset(dataset.X, dataset.Y), batch_size=64, shuffle=True)
    x, y_actual = train_data.dataset.tensors

    # load model
    model = make_anfis(x, num_mfs=num_mfs, num_out=1, hybrid=False)
    model.load_state_dict(torch.load(SAVE_WEIGHTS_PATH))
    # make predictions and calculate RMSE
    model.eval()
    y_pred = model(x)
    RMSE = torch.sqrt(torch.pow(y_pred - y_actual, 2)).sum() / torch.numel(y_actual)
    print("Prediction RMSE:", RMSE)
    plot_all_mfs(model, x)
    # calculate time for single forward pass
    sample_x, sample_y = next(iter(train_data))
    start = time.time()
    y_pred = model(sample_x)
    print(f"Forward pass for one sample took {time.time() - start} s")