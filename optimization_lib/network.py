import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

def create_network(input_dim, hidden_layer_sizes, random_state=42):
    """
    Instantiates an MLPClassifier and does a dummy partial_fit to initialize shapes
    and classes, so we can access coefs_ and intercepts_.
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
    
    model.partial_fit(dummy_X, dummy_y, classes=[0, 1])
    return model

def get_weights(model):
    """
    Flattens and concatenates all network weights and biases into a single 1D numpy array.
    """
    flat_params = []
    for coef, intercept in zip(model.coefs_, model.intercepts_):
        flat_params.append(coef.flatten())
        flat_params.append(intercept.flatten())
    return np.concatenate(flat_params)

def set_weights(model, flat_vector):
    """
    Reconstructs and loads the 1D parameter vector back into model.coefs_ and model.intercepts_.
    """
    coefs = []
    intercepts = []
    current_idx = 0
    
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

def fitness_function(weights, model, X, y, metric="f1_macro"):
    """
    Given a flat weight vector, evaluates the model performance on (X, y)
    and returns a value to maximize.
    """
    set_weights(model, weights)
    predictions = model.predict(X)
    
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
