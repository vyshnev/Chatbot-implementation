# Transformer Chatbot for Movie Dialogues

This repository contains the code for a Transformer-based chatbot trained on the Cornell Movie-Dialogs Corpus. The project demonstrates the end-to-end process of building a sequence-to-sequence model, from data preprocessing to model training and deployment with a Gradio web interface.

A key feature of this project is its modern deployment architecture:
* **Code is hosted on GitHub.**
* **The large model checkpoint (~280 MB) is hosted on Hugging Face Hub** and downloaded dynamically at runtime.

This approach keeps the Git repository lightweight and leverages the best platform for each type of artifact.

## Project Status

**Current State:** Experimental / Proof of Concept

The model has been successfully trained and can generate responses. However, due to significant computational and time constraints, the model was not trained for a sufficient number of epochs to achieve high-quality conversational performance.

As a result, the model has converged to a state of "mode collapse," where it frequently produces safe, generic responses like "i don't know." This is a common challenge in chatbot training that typically requires more extensive training and a larger model architecture to overcome.

The primary goal of this repository is to showcase the complete pipeline and architecture rather than a production-ready chatbot.

## Features

**Data Preprocessing**: Scripts to parse and clean the raw Cornell Movie-Dialogs Corpus into question-answer pairs.
**Custom Vocabulary**: Creation of a word-to-index mapping from the corpus, handling unknown words and special tokens.
**Transformer Model**: A complete implementation of the Transformer architecture in PyTorch.
**Dynamic Model Loading**: The application automatically downloads the model checkpoint from the Hugging Face Hub, ensuring the Git repository remains small.
**Gradio Web UI**: A user-friendly web interface (app.py) to interact with the trained model in real-time.

## Model Architecture & Hyperparameters

### Model Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Transformer | Encoder-Decoder architecture. |
| **Embedding Dimension** (`d_model`) | 512 | The main dimension for embeddings and layers. |
| **Attention Heads** (`n_head`) | 2 | Number of parallel attention mechanisms. |
| **Encoder Layers** | 2 | Number of layers in the encoder stack. |
| **Decoder Layers** | 2 | Number of layers in the decoder stack. |
| **Feed-Forward Dimension** | 512 | Hidden layer size in the FFN. |
| **Dropout Rate** | 0.2 | Regularization dropout probability. |
| **Vocabulary Size** | 15,729 | Total unique words in the vocabulary. |


| Training Detiails | Parameters used |
| :--- | :--- |
| **Optimizer** | Adam with AdamWarmup Scheduler |
| **Loss Function** | KL Divergence with Label Smoothing |
| **Epochs Trained** | 10 |
| **Final Training Loss** | ~30 |

## How to Run

### 1. Clone the Repository
```
git clone https://github.com/vyshnev/Chatbot-implementation.git
cd Chatbot-implementation
```
### 2. Install Dependencies

```
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install torch gradio huggingface_hub
```

### 3. Run the Gradio App

To start the chatbot interface, simply run the `app.py` script:

```
python app.py
```

Note: The first time you run the script, it will automatically download the ~280MB model checkpoint from the Hugging Face Hub (vyshnev/Chatbot-implementation). This might take a few moments depending on your internet connection. Subsequent runs will use the cached file, so they will be much faster.

## Future Work & Improvements
The model's performance can be significantly improved with further training and experimentation. Key areas for future work include:

1. Extended Training: Training the model for many more epochs (50-100+) is the most critical step to lower the loss and move beyond generic responses.
2. Larger Model Architecture: Increasing the number of layers (num_encoder_layers to 6+) and attention heads (n_head to 8) would give the model more capacity to learn complex language patterns.
3. Advanced Decoding: Implementing beam search or nucleus sampling in the evaluate function to generate more diverse and contextually appropriate responses.
4. Hyperparameter Tuning: Experimenting with different learning rates, dropout values, and optimizer settings.
5. Use a Subword Tokenizer: The current word-level tokenizer is limited by a fixed vocabulary. Implementing a subword tokenizer like Byte-Pair Encoding (BPE) or WordPiece would allow the model to handle any word, reduce the vocabulary size, and improve generalization.

## Acknowledgments
* The model is trained on the Cornell Movie-Dialogs Corpus.
* The Transformer architecture is based on the paper Attention Is All You Need by Vaswani et al.
