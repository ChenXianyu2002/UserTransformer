import torch.nn.utils.rnn as rnn_utils
from torch.utils import data


class MyData(data.Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item]


def collate_fn(train_data):
    train_data.sort(key=lambda _data: len(_data), reverse=True)
    data_length = [len(_data) for _data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, data_length


s = MyData([1, 2, 3])
s_d=data.DataLoader(s)
