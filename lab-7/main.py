import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(
        root="./data",
        transform=lambda x: np.array(x).flatten(),
        download=True,
        train=is_train,
    )
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = np.array(train_X)
test_X = np.array(test_X)

train_Y = np.array(train_Y)
test_Y = np.array(test_Y)


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def normalize_data(data):
    return data / 255.0


train_X = normalize_data(train_X)
test_X = normalize_data(test_X)

train_Y_encoded = one_hot_encode(train_Y)
test_Y_encoded = one_hot_encode(test_Y)


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A1, A2


def compute_gradients(X, A1, Y_true, Y_pred, W2):
    m = X.shape[0]
    dZ2 = Y_pred - Y_true
    grad_W2 = np.dot(A1.T, dZ2) / m
    grad_b2 = np.mean(dZ2, axis=0)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    grad_W1 = np.dot(X.T, dZ1) / m
    grad_b1 = np.mean(dZ1, axis=0)

    return grad_W1, grad_b1, grad_W2, grad_b2


def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def evaluate_accuracy(X, Y, W1, b1, W2, b2):
    _, y_pred = forward_propagation(X, W1, b1, W2, b2)
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(Y, axis=1)
    return np.mean(predicted_labels == true_labels)


input_size = 784
hidden_size = 100
output_size = 10
learning_rate = 0.01
decay_factor = 0.1
patience = 5
num_epochs = 500
batch_size = 100

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

best_val_loss = float("inf")
no_improve_epochs = 0

for epoch in range(num_epochs):
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_Y_encoded = train_Y_encoded[indices]

    for start_idx in range(0, train_X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, train_X.shape[0])
        batch_X = train_X[start_idx:end_idx]
        batch_Y = train_Y_encoded[start_idx:end_idx]

        A1, y_pred = forward_propagation(batch_X, W1, b1, W2, b2)
        grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients(
            batch_X, A1, batch_Y, y_pred, W2
        )

        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

    _, train_predictions = forward_propagation(train_X, W1, b1, W2, b2)
    train_loss = cross_entropy_loss(train_predictions, train_Y_encoded)

    _, val_predictions = forward_propagation(test_X, W1, b1, W2, b2)
    val_loss = cross_entropy_loss(val_predictions, test_Y_encoded)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            learning_rate *= decay_factor
            no_improve_epochs = 0
            print(f"Learning rate decayed to {learning_rate:.5f}")

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
    )

train_accuracy = evaluate_accuracy(train_X, train_Y_encoded, W1, b1, W2, b2)
test_accuracy = evaluate_accuracy(test_X, test_Y_encoded, W1, b1, W2, b2)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
