import gradio as gr
import torch
import torch.nn as nn
import json
import math
from huggingface_hub import hf_hub_download
# Step 1: Define Model Architecture
# This architecture MUST EXACTLY match the one used for training.

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_id):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x_embed = self.token_embedding(x)
        return x_embed

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Embeddings(nn.Module):
    def __init__(self, vocab, embed_size, max_len):
        super(Embeddings, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size=len(vocab), embed_size=embed_size, pad_id=vocab["<pad>"])
        self.embed_size = embed_size
        self.pos_embedding = PositionalEmbedding(d_model=embed_size, max_len=max_len + 2)

    def forward(self, x):
        token_embed = self.token_embedding(x) * math.sqrt(self.embed_size)
        pos_embed = self.pos_embedding(x)
        return token_embed + pos_embed

class Transformer(nn.Module):
    def __init__(self, vocab, d_model=512, n_head=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=15) -> None:
        super(Transformer, self).__init__()
        self.vocab = vocab
        d_model = d_model
        n_head = n_head
        num_encoder_layers = num_encoder_layers
        num_decoder_layers = num_decoder_layers
        dim_feedforward = dim_feedforward
        dropout = dropout

        self.input_embedding = Embeddings(vocab, d_model, max_len)

        
        self.transfomrer = torch.nn.Transformer(d_model=d_model, nhead=n_head,
                                                 num_encoder_layers=num_encoder_layers,
                                                 num_decoder_layers=num_decoder_layers,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout, batch_first=True)

        self.proj_vocab_layer = nn.Linear(in_features=d_model, out_features=len(vocab))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.proj_vocab_layer.bias.data.zero_()
        self.proj_vocab_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
        x_enc_embed = self.input_embedding(enc_input.long())
        x_dec_embed = self.input_embedding(dec_input.long())

        src_key_padding_mask = enc_input == self.vocab["<pad>"]
        tgt_key_padding_mask = dec_input == self.vocab["<pad>"]
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))

        # Move masks to the correct device
        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)
        memory_key_padding_mask = memory_key_padding_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        feature = self.transfomrer(src=x_enc_embed, tgt=x_dec_embed,
                                    src_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    tgt_mask=tgt_mask)
        logits = self.proj_vocab_layer(feature)
        return logits


#Step 2: Define Helper and Evaluation Functions

def remove_punc(string):
    punctuations = '''!()-[]{};:'"\\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct += char
    return no_punct.lower()

def evaluate(model, enc_inp, max_len, word_map, reverse_word_map):
    model.eval()
    start_symbol = word_map['<start>']
    end_symbol = word_map['<end>']
    
    dec_inp = torch.LongTensor([start_symbol]).unsqueeze(0).to(device)
    
    for i in range(max_len - 1):
        output = model(enc_inp, dec_inp)
        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        dec_inp = torch.cat([dec_inp, next_token], dim=1)
        if next_token.item() == end_symbol:
            break
            
    tgt_tokens = dec_inp.squeeze(0).tolist()
    sentence = ' '.join([reverse_word_map[token] for token in tgt_tokens if token not in (start_symbol, end_symbol, word_map['<pad>'])])
    
    return sentence

# Step 3: Load Artifacts and Instantiate Model 

# Define Hyperparameters to match the trained model
max_len = 16
d_model = 512
n_head = 2
num_encoder_layers = 2
dim_feedforward = 512
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary
with open('WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)
reverse_word_map = {v: k for k, v in word_map.items()}

# Instantiate the model with the defined architecture
transformer = Transformer(word_map,
                         d_model=d_model,
                         n_head=n_head,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_encoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         max_len=max_len-1).to(device)

# THE NEW, ROBUST LOADING PROCESS 
# 1. Define the Hugging Face Hub repository ID and filename
HF_REPO_ID = "Vyshnev/transformer-chatbot-cornell" 
CHECKPOINT_FILENAME = "new_checkpoint_29.pth" 

print(f"Downloading model from Hugging Face Hub: {HF_REPO_ID}")

# 2. Download the checkpoint file from the Hub.
#    This function downloads the file if it's not cached and returns its local path.
checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILENAME)


# 2. Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# 3. Load the state dictionary into the model
transformer.load_state_dict(checkpoint['model_state_dict'])

# 4. Set the model to evaluation mode
transformer.eval()

print(f"Model loaded successfully from {checkpoint_path}")


# --- Step 4: Create the Chat Function for Gradio ---

def chat(question_string):
    # Pre-process the input string
    clean_question = remove_punc(question_string)
    words = clean_question.split()[:max_len-1] # Truncate to max_len
    
    # Encode the question
    enc_qus = [word_map.get(word, word_map['<unk>']) for word in words]
    question_tensor = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    
    # Get the reply from the model
    reply_sentence = evaluate(transformer, question_tensor, max_len, word_map, reverse_word_map)
    
    return reply_sentence

# --- Step 5: Launch the Gradio Interface ---

iface = gr.Interface(fn=chat,
                     inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
                     outputs="text",
                     title="Transformer Chatbot",
                     description="A chatbot based on the Transformer architecture, trained on the Cornell Movie-Dialogs Corpus.",
                     examples=[["Hello how are you?"], ["Do you eat fruits?"], ["I am happy"]])

if __name__ == "__main__":
    iface.launch()