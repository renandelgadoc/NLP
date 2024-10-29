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

def tokenize(text, vocab_size):
    tokens = list(text.encode("utf-8"))
    # print(f"{[chr(t) for t in tokens]}")

    merges = {}
    num_merges = vocab_size - 256

    for _ in range(num_merges):
        stats = get_stats(tokens)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        new_token_index = 256 + len(merges)
        merges[pair] = new_token_index
        tokens = merge(tokens, pair, new_token_index)

    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    print("Tokens:\n")
    printed_tokens = set()
    for token in tokens:
        token_str = vocab[token].decode("utf-8", errors="replace")
        if token_str not in printed_tokens:
            print(token_str)
            printed_tokens.add(token_str)
# tokenize(text="banan bandana", vocab_size=257)