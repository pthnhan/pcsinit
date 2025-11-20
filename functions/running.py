import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .training import train
from .networks import SimpleNN, FullNN


def create_lag_features(X: np.ndarray, y: np.ndarray, n_lag: int):
    X_lag = np.hstack([X] + [np.roll(X, shift=lag, axis=0) for lag in range(1, n_lag + 1)])
    y_lag = np.hstack([y.reshape(-1, 1)] + [np.roll(y.reshape(-1, 1), shift=lag, axis=0) for lag in range(1, n_lag + 1)])
    valid_idx = np.arange(n_lag, len(y))
    X_lag = X_lag[valid_idx]
    y_lag = y_lag[valid_idx][:, 0]
    return X_lag, y_lag


def mnist_data():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)
    return X_train, X_test, y_train, y_test

def cifar10_data():
    try:
        from keras.datasets import cifar10
    except Exception:
        raise ImportError(
            "cifar10_data() requires the Keras datasets API. Please install keras or modify"
            " this function to use torchvision.datasets.CIFAR10 instead."
        )
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    y_train_tensor = torch.LongTensor(y_train.flatten())
    y_test_tensor = torch.LongTensor(y_test.flatten())
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def prepare_data(X: pd.DataFrame, y: pd.Series, missing: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if missing:
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
        kds = mf.ImputationKernel(X_train_df, random_state=1991)
        kds.mice(10)
        X_train_df = kds.complete_data()
        kds = mf.ImputationKernel(X_test_df, random_state=1991)
        kds.mice(10)
        X_test_df = kds.complete_data()
        X_train = np.asarray(X_train_df)
        X_test = np.asarray(X_test_df)

    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def _zca_whiten(train_data: np.ndarray, test_data: np.ndarray, eps: float = 1e-5):
    X = np.asarray(train_data)
    Y = np.asarray(test_data)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    U, s, _ = np.linalg.svd(cov)
    
    diag = np.diag(1.0 / np.sqrt(s + eps))
    W = U @ diag @ U.T
    X_white = Xc @ W
    Y_white = (Y - mu) @ W
    return X_white, Y_white


def one_run(
    init_type: str,
    X,
    y,
    epochs: int,
    batch_size: int,
    n_layer: int,
    n_frozen_epochs: int = 30,
    missing: bool = False,
    dataname: str | None = None,
    learning_rate = 0.001
):
    if dataname == "mnist":
        X_train, X_test, y_train, y_test = mnist_data()
    elif dataname == "cifar10":
        X_train, X_test, y_train, y_test = cifar10_data()
    else:
        X_train, X_test, y_train, y_test = prepare_data(X, y, missing=missing)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    X_train_cpu = X_train
    X_test_cpu = X_test

    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))

    
    variance_retained = 0.95
    pca = PCA(n_components=variance_retained)
    pca.fit(X_train_cpu)
    n_components = pca.n_components_
    hidden_dim = n_components
    print(f"Number of PCA components: {n_components}")

    
    other_layers = SimpleNN(input_dim=n_components, hidden_dim=hidden_dim, output_dim=output_dim, n_layer=n_layer, init_type=init_type)

    
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    other_layers = other_layers.to(device)

    
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    

    
    print("Training with PCA‑initialized NN (PCsInit) ...")
    pca_init_nn = FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type).to(device)
    pca_init_nn.init_pca_weights(X_train)
    
    optimizer = optim.Adam([{'params': param} for name, param in pca_init_nn.named_parameters() if not name.startswith('fc1')], lr=learning_rate)
    train_losses_pcinit, test_accuracies_pcinit, training_time_pcinit = train(pca_init_nn, train_loader, test_loader, criterion, optimizer, epochs=n_frozen_epochs)
    
    optimizer = optim.Adam(pca_init_nn.parameters(), lr=learning_rate)
    train_losses_pcinit2, test_accuracies_pcinit2, training_time_pcinit2 = train(pca_init_nn, train_loader, test_loader, criterion, optimizer, epochs=epochs - n_frozen_epochs)
    train_losses_pcinit = np.concatenate([train_losses_pcinit, train_losses_pcinit2])
    test_accuracies_pcinit = np.concatenate([test_accuracies_pcinit, test_accuracies_pcinit2])
    training_time_pcinit = np.concatenate([training_time_pcinit, training_time_pcinit2])

    
    print("Training with PCA‑initialized NN (PCsInit‑Sub) ...")
    pca_init_nn_sub = FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type).to(device)
    
    np.random.seed(0)
    n_sample = max(int(X_train.shape[0] * 0.2), n_components + 1)
    indices_sub = np.random.choice(X_train.shape[0], n_sample, replace=False)
    X_train_sub = X_train[indices_sub]
    pca_init_nn_sub.init_pca_weights(X_train_sub)
    optimizer = optim.Adam([{'params': param} for name, param in pca_init_nn_sub.named_parameters() if not name.startswith('fc1')], lr=learning_rate)
    train_losses_pcinit_sub, test_accuracies_pcinit_sub, training_time_pcinit_sub = train(pca_init_nn_sub, train_loader, test_loader, criterion, optimizer, epochs=n_frozen_epochs)
    optimizer = optim.Adam(pca_init_nn_sub.parameters(), lr=learning_rate)
    train_losses_pcinit_sub2, test_accuracies_pcinit_sub2, training_time_pcinit_sub2 = train(pca_init_nn_sub, train_loader, test_loader, criterion, optimizer, epochs=epochs - n_frozen_epochs)
    train_losses_pcinit_sub = np.concatenate([train_losses_pcinit_sub, train_losses_pcinit_sub2])
    test_accuracies_pcinit_sub = np.concatenate([test_accuracies_pcinit_sub, test_accuracies_pcinit_sub2])
    training_time_pcinit_sub = np.concatenate([training_time_pcinit_sub, training_time_pcinit_sub2])

    
    print("Training with PCA‑initialized NN (PCsInit‑Act) ...")
    pca_init_nn_act = FullNN(input_dim, n_components, other_layers, activation='relu', init_type=init_type).to(device)
    pca_init_nn_act.init_pca_weights(X_train)
    optimizer = optim.Adam([{'params': param} for name, param in pca_init_nn_act.named_parameters() if not name.startswith('fc1')], lr=learning_rate)
    train_losses_pcinit_act, test_accuracies_pcinit_act, training_time_pcinit_act = train(pca_init_nn_act, train_loader, test_loader, criterion, optimizer, epochs=n_frozen_epochs)
    optimizer = optim.Adam(pca_init_nn_act.parameters(), lr=learning_rate)
    train_losses_pcinit_act2, test_accuracies_pcinit_act2, training_time_pcinit_act2 = train(pca_init_nn_act, train_loader, test_loader, criterion, optimizer, epochs=epochs - n_frozen_epochs)
    train_losses_pcinit_act = np.concatenate([train_losses_pcinit_act, train_losses_pcinit_act2])
    test_accuracies_pcinit_act = np.concatenate([test_accuracies_pcinit_act, test_accuracies_pcinit_act2])
    training_time_pcinit_act = np.concatenate([training_time_pcinit_act, training_time_pcinit_act2])

    
    print("Training PCA‑NN ...")
    
    simple_nn = SimpleNN(input_dim=n_components, hidden_dim=hidden_dim, output_dim=output_dim, n_layer=n_layer, init_type=init_type).to(device)
    optimizer = optim.Adam(simple_nn.parameters(), lr=learning_rate)
    
    pca_prep = PCA(n_components=n_components)
    X_train_pca = pca_prep.fit_transform(X_train_cpu.numpy())
    X_test_pca = pca_prep.transform(X_test_cpu.numpy())
    X_train_pca_tensor = torch.FloatTensor(X_train_pca).to(device)
    X_test_pca_tensor = torch.FloatTensor(X_test_pca).to(device)
    train_losses_pca_nn, test_accuracies_pca_nn, training_time_pca_nn = train(
        simple_nn,
        torch.utils.data.DataLoader(list(zip(X_train_pca_tensor, y_train)), batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(list(zip(X_test_pca_tensor, y_test)), batch_size=batch_size, shuffle=False),
        criterion,
        optimizer,
        epochs=epochs,
    )

    
    print("Training deeper NN baseline ...")
    nn_baseline = FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type).to(device)
    optimizer = optim.Adam(nn_baseline.parameters(), lr=learning_rate)
    train_losses_nn, test_accuracies_nn, training_time_nn = train(nn_baseline, train_loader, test_loader, criterion, optimizer, epochs=epochs)


    

    
    
    print("Training ZCA MLP (NN‑like architecture) ...")
    X_train_np = X_train_cpu.numpy()
    X_test_np = X_test_cpu.numpy()
    X_train_zca, X_test_zca = _zca_whiten(X_train_np, X_test_np)
    X_train_zca_tensor = torch.FloatTensor(X_train_zca).to(device)
    X_test_zca_tensor = torch.FloatTensor(X_test_zca).to(device)
    zca_loader_train = torch.utils.data.DataLoader(list(zip(X_train_zca_tensor, y_train)), batch_size=batch_size, shuffle=True)
    zca_loader_test = torch.utils.data.DataLoader(list(zip(X_test_zca_tensor, y_test)), batch_size=batch_size, shuffle=False)
    mlp_zca = FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type).to(device)
    optimizer = optim.Adam(mlp_zca.parameters(), lr=learning_rate)
    train_losses_zca, test_accuracies_zca, training_time_zca = train(mlp_zca, zca_loader_train, zca_loader_test, criterion, optimizer, epochs=epochs)

    
    res = [
        train_losses_pcinit, test_accuracies_pcinit,
        train_losses_pca_nn, test_accuracies_pca_nn,
        train_losses_nn, test_accuracies_nn,
        train_losses_pcinit_sub, test_accuracies_pcinit_sub,
        train_losses_pcinit_act, test_accuracies_pcinit_act,
        train_losses_zca, test_accuracies_zca,
    ]
    time = [
        training_time_pcinit, training_time_pca_nn, training_time_nn,
        training_time_pcinit_sub, training_time_pcinit_act,
        training_time_zca,
    ]
    return res, time


def plot_results(
    dataname: str,
    mean_all_run: np.ndarray,
    ci_all_run: np.ndarray | None = None,
    fontsize: int = 16,
    output_dir: str = 'output'
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    model_mapping = [
        ('PCsInit', 0),
        ('PCA‑NN', 2),
        ('Raw MLP', 4),
        ('PCsInit‑Sub', 6),
        ('PCsInit‑Act', 8),
        ('ZCA MLP', 10),
    ]
    
    train_colors = {
        'PCsInit': 'b',
        'PCA‑NN': 'c',
        'Raw MLP': 'g',
        'PCsInit‑Sub': 'k',
        'PCsInit‑Act': 'r',
        'ZCA MLP': 'orange',
    }
    test_colors = {
        'PCsInit': 'lime',
        'PCA‑NN': 'magenta',
        'Raw MLP': 'olive',
        'PCsInit‑Sub': 'teal',
        'PCsInit‑Act': 'gold',
        'ZCA MLP': 'indigo',
    }

    epochs = len(mean_all_run[0])
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    lines_loss: list = []
    labels_loss: list = []
    lines_acc: list = []
    labels_acc: list = []
    
    for label, idx in model_mapping:
        loss = mean_all_run[idx]
        color = train_colors[label]
        if ci_all_run is not None:
            ci = ci_all_run[idx]
            ax1.fill_between(range(epochs), loss - ci, loss + ci, color=color, alpha=0.2)
        line_loss, = ax1.plot(range(epochs), loss, color=color, label=f"{label} loss")
        lines_loss.append(line_loss)
        labels_loss.append(f"{label} loss")
    
    for label, idx in model_mapping:
        acc = mean_all_run[idx + 1]
        color = test_colors[label]
        if ci_all_run is not None:
            ci = ci_all_run[idx + 1]
            ax2.fill_between(range(epochs), acc - ci, acc + ci, color=color, alpha=0.2)
        line_acc, = ax2.plot(range(epochs), acc, color=color, label=f"{label} acc")
        lines_acc.append(line_acc)
        labels_acc.append(f"{label} acc")
    
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.set_ylabel('Training Loss', fontsize=fontsize)
    ax2.set_ylabel('Test Accuracy', fontsize=fontsize)
    ax1.grid(True)
    ax2.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    
    
    loss_legend = fig.legend(
        lines_loss,
        labels_loss,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        fontsize=fontsize,
    )
    acc_legend = fig.legend(
        lines_acc,
        labels_acc,
        loc='lower left',
        bbox_to_anchor=(1.05, 0),
        fontsize=fontsize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataname}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_running_time(
    dataname: str,
    mean_time_all_run: np.ndarray,
    fontsize: int = 16,
    output_dir: str = 'output'
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_names = [
        'PCsInit',
        'PCA‑NN',
        'Raw MLP',
        'PCsInit‑Sub',
        'PCsInit‑Act',
        'ZCA MLP',
    ]
    colors = {
        'PCsInit': 'lime',
        'PCA‑NN': 'magenta',
        'Raw MLP': 'olive',
        'PCsInit‑Sub': 'teal',
        'PCsInit‑Act': 'gold',
        'ZCA MLP': 'purple',
    }
    epochs_range = range(1, len(mean_time_all_run[0]) + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, times in zip(model_names, mean_time_all_run):
        
        times = np.where(times < 0, 0, times)
        cumulative_times = np.cumsum(times)
        ax.plot(epochs_range, cumulative_times, color=colors[model], label=model)
    ax.set_xlabel('Epochs', fontsize=fontsize)
    ax.set_ylabel('Cumulative Training Time (seconds)', fontsize=fontsize)
    ax.set_title('Cumulative Training Time per Epoch for Each Model', fontsize=fontsize)
    ax.legend(loc='upper left', fontsize=fontsize)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataname}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
