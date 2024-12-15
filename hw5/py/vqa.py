import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image

# JSON load a file, include exception handling
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON decode error: {file_path}")
        return None
    return data


class VQADataset(Dataset):
    def __init__(self, img_dir, split="train", transform=None):
        questions = load_json_file(
            f"data/OpenEnded_abstract_v002_{split}2017_questions.json")

        if questions is None:
            print("Error loading questions file")
            exit(1)

        annotations = load_json_file(
            f"data/abstract_v002_{split}2017_annotations.json")

        if annotations is None:
            print("Error loading annotations file")
            exit(1)

        self.annotations = annotations["annotations"]
        self.questions = questions["questions"]
        self.img_dir = img_dir
        self.split = split

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        print(f"Processing image {idx}")
        img_idx = str(self.questions[idx]["image_id"]).zfill(12)
        img_path = f"data/{self.img_dir}/abstract_v002_{self.split}2015_{img_idx}.png"
        image = read_image(img_path)

        answers = [annotation["answer"]
                   for annotation in self.annotations[idx]["answers"]]

        return image, answers


class VQAModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_classes):
        super(VQAModel, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + embed_size, num_classes)

    def forward(self, images, questions):
        image_features = self.cnn(images)
        question_embeddings = self.embedding(questions)
        lstm_out, _ = self.lstm(question_embeddings)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((image_features, lstm_out), dim=1)
        outputs = self.fc(combined)
        return outputs


def train():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Replace with actual data loading
    train_dataset = VQADataset(
        img_dir="scene_img_abstract_v002_train2017", split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    vocab_size = 1000  # Example value
    num_classes = 2
    model = VQAModel(embed_size=256, hidden_size=512,
                     vocab_size=vocab_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for image, labels in train_loader:
            # Assuming labels are a list of multiple labels, we need to handle them appropriately
            # Simplified for single label per image
            labels = torch.tensor([label[0] for label in labels])

            outputs = model(image, labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


def main():
    train()


if __name__ == "__main__":
    train()
