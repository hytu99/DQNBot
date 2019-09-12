import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out
