import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from anfis_pytorch import make_anfis, train_anfis_with, plot_all_mfs
import numpy as np

class SineAndDwellDataset(Dataset):
    def __init__(self, data, selected_features=None):
        self.data = data
        self.selected_features = selected_features
        self.X = self.extractX(data, selected_features)
        self.Y = self.extractY(data)


    def extractX(self, data, selected_features):
        x = torch.Tensor(np.array(data[data.columns[:-1]]))
        if selected_features is not None:
            x = x[:, selected_features]
        return x

    def extractY(self, data):
        y = torch.Tensor(np.array(data[data.columns[-1]])).unsqueeze(1)
        return y

    def __getitem__(self, idx):
        if type(idx) != list:
            idx = [idx]
        sample = np.array(self.data.iloc[idx])

        x = torch.as_tensor(sample[:, :-1], dtype=torch.float32)
        y = torch.as_tensor(sample[:, -1], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    SAVE_WEIGHT_PATH = "data/state_dict_model.pt"
    df = pd.read_csv("data/datasets/mx5_normal_conditions/dataset.csv")
    # load dataset
    dataset = SineAndDwellDataset(df)
    train_data = DataLoader(TensorDataset(dataset.X, dataset.Y), batch_size=64, shuffle=True)
    x, y_actual = train_data.dataset.tensors
    # create model
    model = make_anfis(x, num_mfs=3, num_out=1, hybrid=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    def criterion(input, target):  # change the dim and type
        return torch.nn.MSELoss()(input.squeeze(), target.squeeze())
    train_anfis_with(model, train_data, optimizer, criterion, 60, show_plots=True)
    # calculate MSE
    y_pred = model(x)
    RMSE = torch.sqrt(torch.pow(y_pred - y_actual, 2)).sum() / torch.numel(y_actual)
    print("Prediction RMSE:", RMSE)
    plot_all_mfs(model, x)
    # save model
    torch.save(model.state_dict(), SAVE_WEIGHT_PATH)


