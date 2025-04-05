# training/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from MLmodel.gpt_model import create_model
from model.training.utils import process_file, tokenizer
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = process_file(text, self.tokenizer, self.max_length)
        return {'input_ids': input_ids[0]}

def train(model, dataloader, optimizer, criterion, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids[:, 1:].contiguous()
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output[:, :-1].contiguous().view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
        torch.save(model.state_dict(), f"gpt_model_epoch_{epoch + 1}.pt")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(vocab_size=len(tokenizer), d_model=512, nhead=8, num_layers=12, dim_feedforward=2048).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Example texts (replace with your own dataset)
    texts = ["Electrical engineering problem example", "Programming challenge code", "Civil engineering design", "System Admin",]
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    train(model, dataloader, optimizer, criterion, device)

if __name__ == "__main__":
    main()
