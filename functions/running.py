import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .training import train
from .networks import SimpleNN, FullNN
import matplotlib.pyplot as plt
import miceforest as mf
import os


def create_lag_features(X, y, n_lag):
    X_lag = np.hstack([X] + [np.roll(X, shift=lag, axis=0) for lag in range(1, n_lag + 1)])
    y_lag = np.hstack([y.reshape(-1, 1)] + [np.roll(y.reshape(-1, 1), shift=lag, axis=0) for lag in range(1, n_lag + 1)])
    valid_idx = np.arange(n_lag, len(y))
    X_lag = X_lag[valid_idx]
    y_lag = y_lag[valid_idx][:, 0]  # Only keep the original target values
    return X_lag, y_lag

def mnist_data():
    # import mnist data
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

def prepare_data(X, y, missing = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if missing:
        X_train = pd.DataFrame(X_train, columns = X.columns)
        X_test = pd.DataFrame(X_test, columns = X.columns)
        kds = mf.ImputationKernel(X_train, random_state=1991)
        kds.mice(10) # Run the MICE algorithm for 5 iterations
        X_train = kds.complete_data()
        kds = mf.ImputationKernel(X_test, random_state=1991)
        kds.mice(10) # Run the MICE algorithm for 5 iterations
        X_test = kds.complete_data()
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def one_run(init_type, X, y, epochs, batch_size, n_layer, n_frozen_epochs = 30, missing = False, dataname=None):
    if dataname == "mnist":
        X_train, X_test, y_train, y_test = mnist_data()
    else:
        X_train, X_test, y_train, y_test = prepare_data(X,y, missing = missing)
    
    input_dim = X_train.shape[1]  # Number of features

    output_dim = len(np.unique(y_train))  # Number of classes (for Iris dataset)
    variance_retained = .95
    pca = PCA(n_components=variance_retained)
    pca.fit(X_train)
    n_components = pca.n_components_
    print(n_components)
    hidden_dim = n_components  # Hidden layer size

    other_layers = SimpleNN(input_dim=n_components, hidden_dim=hidden_dim, output_dim=output_dim, n_layer = n_layer, init_type = init_type)

    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01

    # *********** Neural Network with PCA-initialized weights *******************************************************************
    print("Training with PCA-initialized NN...")
    pca_init_nn = FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type)
    pca_init_nn.init_pca_weights(X_train)  # Initialize weights with PCA components

    # train on everything except the first layer
    optimizer = optim.Adam([{'params': param} for name, param in pca_init_nn.named_parameters() if not name.startswith('fc1')],
                            lr=learning_rate)
    train_losses_pcinit, test_accuracies_pcinit, training_time_pcinit = train(pca_init_nn, train_loader, test_loader, criterion, optimizer, epochs=n_frozen_epochs)

    # train the complete network
    optimizer = optim.Adam(pca_init_nn.parameters(), lr=learning_rate)
    train_losses_pcinit2, test_accuracies_pcinit2, training_time_pcinit2 = train(pca_init_nn, train_loader, test_loader, criterion, optimizer, epochs=epochs-n_frozen_epochs)
    train_losses_pcinit = np.concatenate([train_losses_pcinit,train_losses_pcinit2])
    test_accuracies_pcinit = np.concatenate([test_accuracies_pcinit, test_accuracies_pcinit2])
    training_time_pcinit = np.concatenate([training_time_pcinit, training_time_pcinit2])

    # *********** Neural Network with PCA-initialized weights with sub-sample *******************************************************************
    print("Training with PCA-initialized NN...")
    pca_init_nn_sub =  FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type)
    np.random.seed(0)
    n_sample = max(int(X_train.shape[0] * 0.2), n_components + 1)
    X_train_sub_sample = X_train[np.random.choice(X_train.shape[0], n_sample, replace=False)]
    pca_init_nn_sub.init_pca_weights(X_train_sub_sample)  # Initialize weights with PCA components

    # train on everything except the first layer
    optimizer = optim.Adam([{'params': param} for name, param in pca_init_nn_sub.named_parameters() if not name.startswith('fc1')], lr=learning_rate)
    train_losses_pcinit_sub, test_accuracies_pcinit_sub, training_time_pcinit_sub = train(pca_init_nn_sub, train_loader, test_loader, criterion, optimizer, epochs=n_frozen_epochs)

    # train the complete network
    optimizer = optim.Adam(pca_init_nn_sub.parameters(), lr=learning_rate)
    train_losses_pcinit_sub2, test_accuracies_pcinit_sub2, training_time_pcinit_sub2 = train(pca_init_nn_sub, train_loader, test_loader, criterion, optimizer, epochs=epochs-n_frozen_epochs)
    train_losses_pcinit_sub = np.concatenate([train_losses_pcinit_sub,train_losses_pcinit_sub2])
    test_accuracies_pcinit_sub = np.concatenate([test_accuracies_pcinit_sub, test_accuracies_pcinit_sub2])
    training_time_pcinit_sub = np.concatenate([training_time_pcinit_sub, training_time_pcinit_sub2])

    # *********** Neural Network with PCA-initialized weights with Activation *******************************************************************
    print("Training with PCA-initialized NN...")
    pca_init_nn_kernel = FullNN(input_dim, n_components, other_layers, activation='relu', init_type=init_type)
    pca_init_nn_kernel.init_pca_weights(X_train)  # Initialize weights with PCA components

    # train on everything except the first layer
    optimizer = optim.Adam([{'params': param} for name, param in pca_init_nn_kernel.named_parameters() if not name.startswith('fc1')],
                            lr=learning_rate)
    train_losses_pcinit_ker, test_accuracies_pcinit_ker, training_time_pcinit_ker = train(pca_init_nn_kernel, train_loader, test_loader, criterion, optimizer, epochs=n_frozen_epochs)

    # train the complete network
    optimizer = optim.Adam(pca_init_nn_kernel.parameters(), lr=learning_rate)
    train_losses_pcinit_ker2, test_accuracies_pcinit_ker2, training_time_pcinit_ker2 = train(pca_init_nn_kernel, train_loader, test_loader, criterion, optimizer, epochs=epochs-n_frozen_epochs)
    train_losses_pcinit_ker = np.concatenate([train_losses_pcinit_ker,train_losses_pcinit_ker2])
    test_accuracies_pcinit_ker = np.concatenate([test_accuracies_pcinit_ker, test_accuracies_pcinit_ker2])
    training_time_pcinit_ker = np.concatenate([training_time_pcinit_ker, training_time_pcinit_ker2])

    # *********** Train a simple neural network on Principal Components *******************************************************
    simple_nn = other_layers
    optimizer = optim.Adam(simple_nn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Apply PCA to training data
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train.numpy())
    X_test_pca = pca.transform(X_test.numpy())
    X_train_pca, X_test_pca = torch.FloatTensor(X_train_pca), torch.FloatTensor(X_test_pca)

    train_losses_pcaffw, test_accuracies_pcaffw, training_time_pcaffw = train(simple_nn,
                                                        torch.utils.data.DataLoader(list(zip(X_train_pca, y_train)), batch_size=batch_size, shuffle=True),
                                                        torch.utils.data.DataLoader(list(zip(X_test_pca, y_test)), batch_size=batch_size, shuffle=False),
                                                        criterion, optimizer, epochs=epochs)

    #*********** train the network with n_layer + 1 layers ******************************************************************
    relu_nn =  FullNN(input_dim, n_components, other_layers, activation='none', init_type=init_type)
    optimizer = optim.Adam(relu_nn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_losses_fnn, test_accuracies_fnn, training_time_fnn = train(relu_nn, train_loader, test_loader, criterion, optimizer, epochs=epochs)

    res = [train_losses_pcinit, test_accuracies_pcinit, train_losses_pcaffw, test_accuracies_pcaffw, train_losses_fnn, test_accuracies_fnn, train_losses_pcinit_sub, test_accuracies_pcinit_sub, train_losses_pcinit_ker, test_accuracies_pcinit_ker]
    time = [training_time_pcinit, training_time_pcaffw, training_time_fnn, training_time_pcinit_sub, training_time_pcinit_ker]
    return res, time

def plot_results(dataname, mean_all_run, fontsize=16, output_dir = 'output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_losses_pcinit, test_accuracies_pcinit = mean_all_run[0], mean_all_run[1]
    train_losses_pcaffw, test_accuracies_pcaffw = mean_all_run[2], mean_all_run[3]
    train_losses_fnn, test_accuracies_fnn = mean_all_run[4], mean_all_run[5]
    train_losses_pcinit_sub, test_accuracies_pcinit_sub = mean_all_run[6], mean_all_run[7]
    train_losses_pcinit_ker, test_accuracies_pcinit_ker = mean_all_run[8], mean_all_run[9]
    epochs = len(train_losses_pcinit)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define a list of colors for variety
    colors_train = ['b-', 'k-', 'r-', 'c-', 'g-', 'm-', 'y-', 'orange', 'brown', 'purple']
    colors_test = ['lime', 'teal', 'gold', 'magenta', 'olive', 'navy', 'pink', 'indigo', 'grey', 'cyan']

    # Plot training losses
    ax1.plot(range(epochs), train_losses_pcinit, colors_train[0], label='PCsInit loss')  # Blue
    ax1.plot(range(epochs), train_losses_pcinit_sub, colors_train[1], label='PCsInit-Sub loss')  # Black
    ax1.plot(range(epochs), train_losses_pcinit_ker, colors_train[2], label='PCsInit-Act loss')  # Red
    ax1.plot(range(epochs), train_losses_pcaffw, colors_train[3], label='PCA-NN loss')  # Cyan
    ax1.plot(range(epochs), train_losses_fnn, colors_train[4], label='NN loss')  # Green
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.set_ylabel('Training Loss', fontsize=fontsize)
    ax1.legend(loc='lower right', fontsize=fontsize)
    ax1.grid(True)

    # Plot test accuracies
    ax2 = ax1.twinx()
    ax2.plot(range(epochs), test_accuracies_pcinit, colors_test[0], label='PCsInit acc')  # Lime
    ax2.plot(range(epochs), test_accuracies_pcinit_sub, colors_test[1], label='PCsInit-Sub acc')  # Teal
    ax2.plot(range(epochs), test_accuracies_pcinit_ker, colors_test[2], label='PCsInit-Act acc')  # Gold
    ax2.plot(range(epochs), test_accuracies_pcaffw, colors_test[3], label='PCA-NN acc')  # Magenta
    ax2.plot(range(epochs), test_accuracies_fnn, colors_test[4], label='NN acc')  # Olive
    ax2.set_ylabel('Test Accuracy', fontsize=fontsize)
    ax2.legend(loc='upper left', fontsize=fontsize)
    ax2.grid(False)

    ax2.set_xlabel('Epochs', fontsize=fontsize)
    ax2.set_ylabel('Testing Accuracy (%)', fontsize=fontsize)

    ax2.legend(loc='upper right', fontsize=fontsize)
    ax2.grid(False)

    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/'+dataname+'.png', dpi = 300, bbox_inches='tight')
    plt.show()


def plot_running_time(dataname, mean_time_all_run, fontsize=16, output_dir='output'):
    # Example data structure
    training_time = {
        "PCsInit": mean_time_all_run[0],
        "PCA-NN": mean_time_all_run[1],
        "NN": mean_time_all_run[2],
        "PCsInit-Sub": mean_time_all_run[3],
        "PCsInit-Act": mean_time_all_run[4],
    }
    colors = {
        "PCsInit": 'lime',
        "PCA-NN": 'magenta',
        "NN": 'olive',
        "PCsInit-Sub": 'teal',
        "PCsInit-Act": 'gold',
    }

    epochs = range(1, len(mean_time_all_run[0]) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))


    for model, times in training_time.items():
        times = np.where(times < 0, 0, times)
        cumulative_times = np.cumsum(times)
        ax.plot(epochs, cumulative_times, colors[model], label=model)

    ax.set_xlabel('Epochs', fontsize=fontsize)
    ax.set_ylabel('Cumulative Training Time (seconds)', fontsize=fontsize)
    ax.set_title("Cumulative Training Time per Epoch for Each Model", fontsize=fontsize)

    ax.legend(loc='upper left', fontsize=fontsize)
    ax.grid(True)

    plt.savefig(f'{output_dir}/{dataname}.png', dpi=300, bbox_inches='tight')
    plt.show()

