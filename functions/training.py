import torch
import time


def train(model, train_loader, test_loader, criterion, optimizer, epochs=100):
    train_losses = []
    test_accuracies = []
    training_time = []
    for epoch in range(epochs):
        t_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        t_end = time.time()
        training_time.append(t_end - t_start)

        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)


        if (epochs % 10 == 0):
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Testing Accuracy: {test_accuracy:.4f}, Training Time: {sum(training_time):.4f}')


    return train_losses, test_accuracies, training_time
