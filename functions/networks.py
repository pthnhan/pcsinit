import torch
import torch.nn as nn
import copy
from sklearn.decomposition import PCA


class LinearWithActivation(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, activation='relu', **activation_kwargs):
        super().__init__(in_features, out_features, bias)
        self.activation = self._get_activation(activation, **activation_kwargs)

    def _get_activation(self, activation, **kwargs):
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'none': None,
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activation_map.keys())}")
        activation_class = activation_map[activation]
        return activation_class(**kwargs) if activation_class else None

    def forward(self, x):
        x = super().forward(x)
        return self.activation(x) if self.activation else x


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, init_type='xavier', activation='relu'):
        super(SimpleNN, self).__init__()
        self.fc1 = LinearWithActivation(input_dim, hidden_dim, activation=activation)
        self.hidden_layers = nn.ModuleList([
            LinearWithActivation(hidden_dim, hidden_dim, activation=activation) for _ in range(n_layer - 1)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self._init_weights(init_type)

    def _init_weights(self, init_type):
        initializer = self._get_initializer(init_type)
        initializer(self.fc1.weight)
        for layer in self.hidden_layers:
            initializer(layer.weight)
        initializer(self.fc_out.weight)

    def _get_initializer(self, init_type):
        init_map = {
            'xavier': nn.init.xavier_uniform_,
            'he': nn.init.kaiming_uniform_,
            'orthogonal': nn.init.orthogonal_,
            'uniform': nn.init.uniform_,
        }
        if init_type not in init_map:
            raise ValueError(f"Unknown init_type: {init_type}")
        return init_map[init_type]

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.fc_out(x)


class FullNN(nn.Module):
    def __init__(self, input_dim, n_components, other_layers, activation='none', init_type='xavier', variance_retained=None):
        super(FullNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = LinearWithActivation(input_dim, n_components, bias=False, activation=activation)
        self.other_layers = copy.deepcopy(other_layers)
        self.n_components = n_components
        self.variance_retained = variance_retained
        self._init_fc1(init_type)

    def _init_fc1(self, init_type):
        initializer = self._get_initializer(init_type)
        initializer(self.fc1.weight)

    def _get_initializer(self, init_type):
        init_map = {
            'xavier': nn.init.xavier_uniform_,
            'he': nn.init.kaiming_uniform_,
            'orthogonal': nn.init.orthogonal_,
            'uniform': nn.init.uniform_,
        }
        if init_type not in init_map:
            raise ValueError(f"Unknown init_type: {init_type}")
        return init_map[init_type]

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        return self.other_layers(x)

    def init_pca_weights(self, X_batch):
        pca = PCA(n_components=self.variance_retained if self.n_components is None else self.n_components)
        pca.fit(X_batch.cpu().detach().numpy())
        self.fc1.weight.data = torch.Tensor(pca.components_).to(self.fc1.weight.device)
        print(f'Number of PCA components: {pca.n_components_}')
