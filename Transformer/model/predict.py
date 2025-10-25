# predict.py
import torch
from config import max_gen_tokens

def predict(model, prompt, vocab, max_words=max_gen_tokens):
    """
    Generate tokens autoregressively from a given prompt.

    Args:
        model: Trained Transformer model
        prompt (list[str]): Starting tokens (already tokenized)
        vocab (dict): Mapping of word → token id
        max_words (int): Maximum number of tokens to generate
    Returns:
        list[str]: Generated token sequence (excluding <bos>)
    """

    model.eval()
    id_to_word = {i: w for w, i in vocab.items()}

    # Turn prompt words into token IDs
    context = torch.tensor([[vocab[w] for w in prompt]], dtype=torch.long)

    # Decoder starts with <bos>
    generated = torch.tensor([[vocab['<bos>']]], dtype=torch.long)

    for _ in range(max_words):
        with torch.no_grad():
            logits = model(context, generated)  # [B, seq_len, vocab_size]
            next_word_logits = logits[:, -1, :]  # last step logits
            next_word_id = torch.argmax(next_word_logits, dim=-1)  # greedy decode

        # Add predicted token
        generated = torch.cat([generated, next_word_id.unsqueeze(1)], dim=1)

        # Stop on <eos>
        if next_word_id.item() == vocab.get('<eos>', -1):
            break

    # Convert back to readable words (skip <bos>)
    words = [id_to_word[idx.item()] for idx in generated[0]][1:]
    return words
