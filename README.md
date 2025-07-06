🧠 QuickDraw Image Classifier — Neural Network from Scratch
📝 Overview

This project implements a deep neural network (DNN) from scratch using Python and NumPy to classify hand-drawn images from the Google Quick, Draw! dataset. The model distinguishes between five object categories:
Class	Example
🎧 Headphones	
🔷 Hexagon	
🍕 Pizza	
🪣 Bucket	
🕒 Clock	

The project aims to demystify neural networks by building everything manually—no TensorFlow or PyTorch!
✨ Features

    ✅ Built entirely with NumPy (no deep learning libraries)

    ✅ Loads & caches 5 QuickDraw classes (5,000 samples/class)

    ✅ Preprocessing: grayscale normalization + one-hot labels

    ✅ Architecture: MLP with ReLU & Softmax

    ✅ Optimization: Adam, L2 regularization, dropout

    ✅ Supports inference on custom hand-drawn images

🔧 Installation

git clone https://github.com/your-username/quickdraw-classification.git
cd quickdraw-classification
pip install numpy requests

🚀 Usage
🏁 Train the Model

python main.py

    ownloads and preprocesses data (once).

    Trains MLP for 20 epochs on 80% of data.

    Evaluates on remaining 20%.

    Saves trained parameters to QuickDraw_model_params.pkl.

🧪 Inference on Custom Image

python inference.py path/to/your/image.png

Image must be 28x28, grayscale, and follow QuickDraw style (black ink on white background).

Outputs predicted class + confidence scores.


📊 Results
Metric	Value
Training Accuracy	95%
Test Accuracy	93%
Loss Function	Cross-Entropy
Optimizer	Adam
Regularization	L2 + Dropout
Dropout Rate	0.2

    ⚠️ These results are comparable to official MLP baselines on similar QuickDraw subsets.

🏗️ Model Architecture

Input Layer      → 784 neurons (28×28 flattened)
Hidden Layer 1   → 300 neurons (ReLU activation)
Hidden Layer 2   → 100 neurons (ReLU activation)
Output Layer     → 5 neurons (Softmax activation)

🔍 Training Setup
Parameter	Value
Epochs	20
Batch Size	64
Learning Rate	0.001
Regularization λ	0.001
Dropout	0.2
Dataset Split	80% train / 20% test

📂 Project Structure

quickdraw-classification/
├── test_doodle.py/                       # Cached .npy datasets
├── train_model.py                     # Main training script
├── load_dataset.py                # Custom image classifier
├── QuickDraw_model_params.pkl  # Saved model weights
└── README.md                   # This file


📈 Future Enhancements

⚙️ Add Convolutional Neural Network (CNN) version

🎨 Visualize intermediate activations & filters

🧩 Support 20+ QuickDraw classes

🕒 Leverage temporal stroke data for sequence models

    🌍 Interactive web demo (Gradio or Flask)

🙌 Contributing

    Fork the repo

    Create a new branch (git checkout -b feature/my-feature)

    Commit your changes (git commit -m 'Add feature')

    Push (git push origin feature/my-feature)

    Open a Pull Request

📚 References

    Quick, Draw! Dataset (Google)

    QuickDraw Dataset GitHub

    DL Team @ TelecomBCN

    [Building Neural Networks from Scratch - Sanddex, Vizura]

⚖️ License

This project is licensed under the MIT License.
🙏 Acknowledgments

Thanks to the creators of the QuickDraw dataset and the many open-source contributors who inspired this from-scratch approach to neural networks.


