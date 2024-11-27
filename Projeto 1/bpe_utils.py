vocab = {idx: bytes([idx]) for idx in range(256)}

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def train(text):
    tokens = list(text.encode("utf-8"))
    tokens = list(map(int, tokens))
    vocab_size = 276
    num_merges = vocab_size - 256
    ids = list(tokens)

    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges

def encode(text, merges):
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

def decode(ids, merges):
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    decoded_tokens = [vocab[idx].decode("utf-8", errors="replace") for idx in ids]
    return decoded_tokens

