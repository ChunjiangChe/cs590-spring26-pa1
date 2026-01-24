import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms
import math
import random

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    """TODO: Your code here"""
    W_q, W_k, W_v, W_1, b_1, W_2, b_2 = nodes
    Q = ad.matmul(X, ad.broadcast(W_q, input_shape=(seq_length, model_dim), target_shape=(batch_size, seq_length, model_dim)))  # (batch_size, seq_length, model_dim)
    K = ad.matmul(X, ad.broadcast(W_k, input_shape=(seq_length, model_dim), target_shape=(batch_size, seq_length, model_dim)))  # (batch_size, seq_length, model_dim)
    V = ad.matmul(X, ad.broadcast(W_v, input_shape=(seq_length, model_dim), target_shape=(batch_size, seq_length, model_dim)))  # (batch_size, seq

    attention_scores = ad.matmul(Q, ad.transpose(K, -2, -1))  # (batch_size, seq_length, seq_length)
    attention_scores_scaled = ad.div_by_const(attention_scores, np.sqrt(model_dim))
    attention_weights = ad.matmul(ad.softmax(attention_scores_scaled, dim=-1), V)  # (batch_size, seq_length, seq_length)

    attention_normalized_1 = ad.layernorm(attention_weights, normalized_shape=[model_dim], eps=eps)  # (batch_size, seq_length, seq_length)
    feed_forward_out = ad.matmul(attention_normalized_1, ad.broadcast(W_1, input_shape=[model_dim, model_dim], target_shape=[batch_size, model_dim, model_dim])) + ad.broadcast(b_1, input_shape=[model_dim], target_shape=[batch_size, seq_length, model_dim])
    feed_forward_out = ad.relu(feed_forward_out)
    feed_forward_out = ad.matmul(feed_forward_out, ad.broadcast(W_2, input_shape=[model_dim, num_classes], target_shape=[batch_size, model_dim, num_classes])) + ad.broadcast(b_2, input_shape=[num_classes], target_shape=[batch_size, seq_length, num_classes])

    attention_normalized_2 = ad.layernorm(feed_forward_out, normalized_shape=[num_classes], eps=eps)
    output = ad.mean(feed_forward_out, dim=(1, ), keepdim=False)
    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""
    p = ad.log(ad.softmax(Z))
    res = ad.mean(-1.0 * ad.sum_op(ad.mul(y_one_hot, p), dim=(-1,), keepdim=False), dim=(-1,), keepdim=False)
    return res


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    
    for i in range(num_batches):
        print(f"Processing batch {i}/{num_batches}")

        if random.random() < 0.5:
            continue

        # Your logic here
        # Get the mini-batch data
        start_idx = i * batch_size
        # if start_idx + batch_size > num_examples:
        #     continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        # TODO: Your code here
        _, loss, *gradients = f_run_model([X_batch, y_batch] + model_weights)

        max_norm=1.0
        total_norm = torch.sqrt(sum(torch.sum(g**2) for g in gradients))
    
        if total_norm > max_norm:
            scale_factor = max_norm / (total_norm + 1e-6)  
            gradients = [g * scale_factor for g in gradients]

        
        # Update weights and biases
        # TODO: Your code here
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)
        for w, w_grad in zip(model_weights, gradients):
            w -= lr * w_grad

        # Accumulate the loss
        # TODO: Your code here
        total_loss += loss.item()
    # Compute the average loss
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    # Set up model params

    # TODO: Tune your hyperparameters here
    W_embed = ad.Variable("W_embed")
    W_Q = ad.Variable("W_Q")
    W_K = ad.Variable("W_K")
    W_V = ad.Variable("W_V")
    W_1 = ad.Variable("W_1")
    b_1 = ad.Variable("b_1")
    W_2 = ad.Variable("W_2")
    b_2 = ad.Variable("b_2")
    model_params = [W_Q, W_K, W_V, W_1, b_1, W_2, b_2]
    # Hyperparameters
    seq_length = max_len
    input_dim = seq_length  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 10
    batch_size = 50
    lr = 0.02

    # TODO: Define the forward graph.
    x = ad.Variable('x')

    y_predict: ad.Node = transformer(x, nodes=model_params, model_dim=model_dim, seq_length=seq_length, eps=float(eps), batch_size=batch_size, num_classes=num_classes) # TODO: The output of the forward pass
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # TODO: Construct the backward graph.
    training_inputs = [x, y_groundtruth] + model_params
    inference_inputs = [x] + model_params

    # TODO: Create the evaluator.
    grads: List[ad.Node] = ad.gradients(loss, nodes=model_params) # TODO: Define the gradient nodes here
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays

    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()


    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                # TODO: Fill in the mapping from variable to tensor
                node: inp for node, inp in zip(training_inputs, model_weights)

            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run(
                # TODO: Fill in the mapping from variable to tensor
                input_values={
                    node: inp for node, inp in zip(inference_inputs, [X_batch] + model_weights)
                }
            )
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    W_Q_val = torch.from_numpy(W_Q_val).to(torch.float64)
    W_K_val = torch.from_numpy(W_K_val).to(torch.float64)
    W_V_val = torch.from_numpy(W_V_val).to(torch.float64)
    W_1_val = torch.from_numpy(W_1_val).to(torch.float64)
    W_2_val = torch.from_numpy(W_2_val).to(torch.float64)
    b_1_val = torch.from_numpy(b_1_val).to(torch.float64)
    b_2_val = torch.from_numpy(b_2_val).to(torch.float64)
    model_weights: List[torch.Tensor] = [W_Q_val, W_K_val, W_V_val, W_1_val, b_1_val, W_2_val, b_2_val] # TODO: Initialize the model weights here
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        # subset_size = 1000
        # indices = np.random.choice(len(X_train), subset_size, replace=False)
        # X_train = X_train[indices]
        # y_train = y_train[indices]

        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")