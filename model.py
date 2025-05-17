import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

# ---- Config ----
SEQ_LEN = 4
EMBED_DIM = 32
EPOCHS = 300
LR = 0.01

# ---- Sample Paragraph ----
paragraph = """ It happened in an instant.

To be precise, it happened in somewhere between a picosecond and a second. If you had tried to say the words “Sarglop’s Ramen Noodles” in the time it took to happen, it would be over before you finished taking the little breath that humans take before they start speaking. Not even the “s” from “Sarglop’s” would be heard, and anyone wondering what you were going to say would be left with no choice but to guess. Even if, by some cosmic coincidence, they guessed that you had been about to say the name of a noodle, they would have almost no chance of knowing which one, and would probably guess a less thick and less slurpy noodle than Sarglop’s.

So, to tell you about it slowly would be inappropriate, given the shocking speed at which it occurred.

In fact, Aggomalda’s Universal Ratio, which states that the length of a story may not exceed ten times the length of the event it is describing, dictates that there can be no build up at all. Therefore, at this moment, I can’t tell you why a personal assistant was in space. I cannot even explain how he came to be eating a tuna fish sandwich that wasn’t meant for him.

I know you’re used to being told a great many things in books, written in long sentences with many details, but it simply wouldn’t be right, given how quickly it happened, to say anything but this:

One moment Jackson Fickle was taking the second bite of a tuna fish sandwich, and the next he was pulled inside a black hole.

Of course, it’s true that I could go back and tell you more about him now, having described the speedy event speedily, but I think you will find that when black holes are involved, the best way backwards is forwards. And as you now know, a black hole is exactly where Jackson found himself after just one-and-a-half bites of someone else's tuna fish sandwich.

He was still chewing when he noticed that his ROC (Roving Observational Craft) was now somewhere that neither he, nor I, can describe to you. And it is important to note that while I chose not to describe Jackson and how he got here in the first place, I am not now choosing to not describe his black hole (which would one day be named ‘Jackson Hole’ after him) because I do not wish to do so, but because it cannot be described. Black holes are, in fact, impossible to describe. The official Universal University of the Universe’s list of things that cannot be described is exactly five items long, and blacks holes are number two on that list. *At the time of this writing, a sixth item, the feeling of having an oncoming sneeze interrupted, is being considered for the list, but has not yet been accepted.

So, to try to describe what Jackson saw and felt inside the black hole would be a waste of time. Instead, I can only share with you the one thing that Jackson remembers clearly about the experience and insists upon to this day: when he came out the other side he had the taste of peppermint in his mouth. And, given that he had been eating a tuna fish sandwich, which is on the opposite side of the UU of U’s spectrum of tastes, you can imagine the drastic nature of his short journey. """

# ---- Tokenize ----
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    vocab = sorted(set(words))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    tokens = [word_to_id[w] for w in words]
    return tokens, word_to_id, id_to_word

tokens, word_to_id, id_to_word = tokenize(paragraph)
VOCAB_SIZE = len(word_to_id)

# ---- Dataset ----
inputs, targets = [], []
for i in range(len(tokens) - SEQ_LEN):
    inputs.append(tokens[i:i+SEQ_LEN])
    targets.append(tokens[i+SEQ_LEN])

x = torch.tensor(inputs)
y = torch.tensor(targets)

# ---- Model ----
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) #query vector what we are looking for
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) #key vector short describition of what we have
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) #value vector actual data
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.to_vocab = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        x = self.embed(x)  # (B, T, D)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / EMBED_DIM**0.5 #calculation of how similar query and key is
        attn_probs = F.softmax(attn_scores, dim=-1) # to covert to probability distribution
        context = torch.matmul(attn_probs, V) #chossing which value is needed

        out = self.out_proj(context)
        pooled = out.mean(dim=1)
        return self.to_vocab(pooled)

model = TinyTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ---- Train ----
for epoch in range(EPOCHS):
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"Final loss: {loss.item():.4f}")

# ---- Save weights ----
def save_bin(tensor, filename):
    tensor.detach().cpu().numpy().astype(np.float32).tofile(filename)

save_bin(model.embed.weight, "embedding.bin")
save_bin(model.q_proj.weight, "W_q.bin")
save_bin(model.k_proj.weight, "W_k.bin")
save_bin(model.v_proj.weight, "W_v.bin")
save_bin(model.out_proj.weight, "W_o.bin")
save_bin(model.out_proj.bias, "b_o.bin")

# ---- Save vocab ----
with open("vocab.txt", "w") as f:
    for word, idx in word_to_id.items():
        f.write(f"{word} {idx}\n")

print("Weights and vocab saved.")
