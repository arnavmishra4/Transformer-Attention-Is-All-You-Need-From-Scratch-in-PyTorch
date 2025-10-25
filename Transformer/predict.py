import torch

def predict(model, input_tokens, vocab, max_words=20, device="cpu"):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}  # reverse vocab: id → token

    input_tensor = torch.tensor([[vocab[w] for w in input_tokens]], dtype=torch.long).to(device)

    decoder_input = torch.tensor([[vocab["<bos>"]]], dtype=torch.long).to(device)

    for _ in range(max_words):
        with torch.no_grad():
            logits = model(input_tensor, decoder_input)  # [B, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]         # last token predictions
            next_token = torch.argmax(next_token_logits, dim=-1)  # greedy decode
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == vocab.get("<eos>", -1):
            break
    predicted_tokens = [inv_vocab[idx.item()] for idx in decoder_input[0]][1:]
    return predicted_tokens
