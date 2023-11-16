import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        out, _ = self.lstm(embedded)
        out = self.fc(out[:, -1, :])
        return out

# Tokenizer and iterator
tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split='train')
tokens = tokenizer(next(iter(train_iter))['text'])
vocab = build_vocab_from_iterator([tokens], specials=["<unk>", "<pad>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# Numericalize the data
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

# Define the model, loss function, and optimizer
input_size = len(vocab)
embedding_dim = 32
hidden_size = 64
output_size = 4  # AG_NEWS has 4 classes
learning_rate = 0.01
num_epochs = 5

model = LSTMModel(input_size, embedding_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader
train_dataloader = DataLoader(to_map_style_dataset(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)

# Training the model
for epoch in range(num_epochs):
    for labels, text, offsets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(text, offsets)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model if needed
# torch.save(model.state_dict(), 'lstm_model.pth')
