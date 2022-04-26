import torch.nn as nn

from .set_transformer import ISAB, PMA, SAB


class MLP(nn.Module):
    activation_classes = {
        'relu': nn.ReLU,
    }
    def __init__(self, dims, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = dims[0]
        for dim in dims[1:-1]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation_classes[activation]())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_attention, n_mlp_layer,
                 n_ind=32, n_head=4, ln=False, activation='relu'):
        super().__init__()

        attention_layers =  [ISAB(in_dim, hidden_dim, n_head, n_ind, ln=ln)]
        attention_layers += [
            ISAB(hidden_dim, hidden_dim, n_head, n_ind, ln=ln)
            for _ in range(n_attention-1)
        ]
        self.attention = nn.Sequential(*attention_layers)
        self.pool = PMA(hidden_dim, n_head, 1, ln=ln)
        self.mlp = MLP([hidden_dim]*n_mlp_layer + [out_dim], activation=activation)

    def forward(self, batch_set_x):
        return self.mlp(self.pool(self.attention(batch_set_x)).squeeze(1))
