from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, sentences: list[str]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def get_dataloader(sentences: list[str], batch_size: int, shuffle=True):
    return DataLoader(
        TextDataset(sentences),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
