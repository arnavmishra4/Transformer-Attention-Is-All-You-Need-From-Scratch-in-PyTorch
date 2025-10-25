import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_json, build_vocab, tokens_to_tensor
from config import train_path, batch_size

class TransformerDataset(Dataset):
    def __init__(self, data_path, vocab=None):
        self.data = load_json(data_path)
        self.vocab = vocab or build_vocab(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src = sample["input"]
        tgt = sample["target"]
        tgt_in = tgt[:-1]
        tgt_out = tgt[1:]

        src_ids = [self.vocab[w] for w in src]
        tgt_in_ids = [self.vocab[w] for w in tgt_in]
        tgt_out_ids = [self.vocab[w] for w in tgt_out]

        return torch.tensor(src_ids), torch.tensor(tgt_in_ids), torch.tensor(tgt_out_ids)


def collate_fn(batch):
    src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_in_padded = torch.nn.utils.rnn.pad_sequence(tgt_in_batch, batch_first=True, padding_value=0)
    tgt_out_padded = torch.nn.utils.rnn.pad_sequence(tgt_out_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_in_padded, tgt_out_padded


def get_dataloader(vocab=None, shuffle=True):
    dataset = TransformerDataset(train_path, vocab=vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader, dataset.vocab
