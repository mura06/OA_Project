import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

def create_network(input_dim, hidden_layer_sizes, random_state=42):
    """
    Instantiates an MLPClassifier and does a dummy partial_fit to initialize shapes
    and classes, so we can access coefs_ and intercepts_.

    Parameters:
        input_dim: The input dimension of the neural network.
        hidden_layer_sizes: The number of hidden layers and their sizes.
        random_state: The random state for reproducibility.
    
    Returns:
        MLPClassifier: The initialized neural network model.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=1,
        random_state=random_state,
        activation='relu',
        solver='adam'
    )
    
    # Initialize shapes with dummy data matching input_dim and binary classification
    dummy_X = np.zeros((2, input_dim))
    dummy_y = np.array([0, 1])
    
    # Call partial_fit to initialize weights and intercepts (shapes) to non-None values
    # We use classes=[0, 1] to explicitly tell it this is a binary classification problem
    model.partial_fit(dummy_X, dummy_y, classes=[0, 1])
    return model

def get_weights(model):
    """
    Flattens and concatenates all network weights and biases into a single 1D numpy array.~
    
    Parameters:
        model: The neural network model.
    
    Returns:
        np.ndarray: The flattened weight vector.
    """
    # Create empty list to store flattened coefficients and intercepts
    flat_params = []

    # Loop through coefficients and intercepts and flatten them
    for coef, intercept in zip(model.coefs_, model.intercepts_):
        flat_params.append(coef.flatten())
        flat_params.append(intercept.flatten())

    # Concatenate flattened coefficients and intercepts
    return np.concatenate(flat_params)

def set_weights(model, flat_vector):
    """
    Reconstructs and loads the 1D parameter vector back into model.coefs_ and model.intercepts_.
    
    Parameters:
        model: The neural network model.
        flat_vector: The flattened weight vector.
    
    Returns:
        None
    """
    # Reconstruct coefficients and intercepts from flattened vector
    coefs = []
    intercepts = []
    current_idx = 0
    
    # Loop through coefficients and intercepts and reshape them
    for i in range(len(model.coefs_)):
        coef_shape = model.coefs_[i].shape
        intercept_shape = model.intercepts_[i].shape
        
        num_coef = np.prod(coef_shape)
        num_intercept = np.prod(intercept_shape)
        
        coef_flat = flat_vector[current_idx : current_idx + num_coef]
        current_idx += num_coef
        
        intercept_flat = flat_vector[current_idx : current_idx + num_intercept]
        current_idx += num_intercept
        
        coefs.append(coef_flat.reshape(coef_shape))
        intercepts.append(intercept_flat.reshape(intercept_shape))
        
    model.coefs_ = coefs
    model.intercepts_ = intercepts

def generate_solution(model, init_method="uniform", random_state=42):
    """
    Generates a random solution (weight vector) for the given model.
    Supports 'uniform' (Xavier/Glorot) and 'normal' (He) initializations.

    Parameters:
        model: The neural network model.
        init_method: The initialization method ('uniform' or 'normal').
        random_state: The random state for reproducibility.
    
    Returns:
        np.ndarray: The flattened weight vector.
    """
    # If random_state is not a Generator, create one
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Create empty list to store newly generated weights
    parts = []
    
    # Loop through layer sizes and generate random weights for each layer
    for i in range(len(model.coefs_)):
        n_in, n_out = model.coefs_[i].shape
        if init_method == "uniform":
            limit = np.sqrt(6.0 / (n_in + n_out))
            parts.append(rng.uniform(-limit, limit, size=n_in * n_out))
            parts.append(rng.uniform(-limit, limit, size=n_out))
        else:
            std = np.sqrt(2.0 / n_in)
            parts.append(rng.normal(0, std, size=n_in * n_out))
            parts.append(rng.normal(0, std, size=n_out))
    
    # Return concatenated newly generated weight vector
    return np.concatenate(parts)

def fitness_function(weights, model, X, y, metric="f1_macro"):
    """
    Given a flat weight vector, evaluates the model performance on (X, y)
    and returns a value to maximize.
    
    Parameters:
        weights: The flattened weight vector.
        model: The neural network model.
        X: The input data.
        y: The target data.
        metric: The metric to evaluate ('f1_macro', 'f1_binary', 'accuracy', 'balanced_accuracy').
    
    Returns:
        float: The fitness value.
    """
    set_weights(model, weights)
    predictions = model.predict(X)
    
    # Return fitness value according to the chosen metric
    if metric == "f1_macro":
        return f1_score(y, predictions, average="macro")
    elif metric == "f1_binary":
        return f1_score(y, predictions, average="binary")
    elif metric == "accuracy":
        return accuracy_score(y, predictions)
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(y, predictions)
    else:
        raise ValueError(f"Unknown metric: {metric}")
