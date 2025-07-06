import numpy as np
import requests
import io
import os

def load_quickdraw_data(num_samples_per_class=5000):
    classes = ['headphones', 'hexagon', 'pizza', 'bucket', 'clock']
    X, y = [], []
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    for class_idx, class_name in enumerate(classes):
        file_path = f"data/{class_name}.npy"
        if os.path.exists(file_path):
            print(f"Loading {class_name} from cache...")
            data = np.load(file_path)
        else:
            print(f"Downloading {class_name}...")
            url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
            response = requests.get(url)
            data = np.load(io.BytesIO(response.content))
            # Save to local file
            np.save(file_path, data)
        
        data = data[:num_samples_per_class]  # Limit samples
        X.append(data)
        y.append(np.full((data.shape[0],), class_idx))
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    # Reshape to (num_samples, 784) since 28x28=784
    X = X.reshape(X.shape[0], -1)
    # One-hot encode labels
    y_one_hot = np.zeros((y.shape[0], len(classes)))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    
    print("Data loading complete.")
    return X, y_one_hot, classes

def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(test_size * X.shape[0])
    train_idx, test_idx = indices[test_size:], indices[:test_size]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]