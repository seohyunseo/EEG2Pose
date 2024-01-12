import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, in_col, out_col, type=None):
        if type == 'Train':
            csv_file = csv_file + '/train.csv'
        elif type == 'Validation':
            csv_file = csv_file + '/validation.csv'
        elif type == 'Test':
            csv_file = csv_file + '/test.csv'

        self.data = pd.read_csv(csv_file)
        self.in_col = in_col
        self.out_col = out_col

        self.input_indices = [in_col.index(channel) for channel in in_col]
        self.output_indices = [out_col.index(keypoint) for keypoint in out_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = torch.tensor(sample[self.in_col].values[self.input_indices], dtype=torch.float32)
        labels = torch.tensor(sample[self.out_col].values[self.output_indices], dtype=torch.float32)

        return features, labels