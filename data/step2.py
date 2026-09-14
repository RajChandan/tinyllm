################
# step1.py

with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("training text : ")
print(text)

chars = sorted(list(set(text)))

vocab_size = len(chars)

print("\n vocabulary : ")

print(chars)

print(" vocabulary size : ", vocab_size)

char_to_id = {char: index for index, char in enumerate(chars)}

id_to_char = {index: char for index, char in enumerate(chars)}

print("Token ID : ")
print(char_to_id)


def encode(text):
    return [char_to_id[ch] for ch in text]


def decode(token_ids):
    return "".join(id_to_char[token_id] for token_id in token_ids)


example = "the cat"

encoded = encode(example)

print("Original")

print(example)

print("encoded : ")
print(encoded)

print("Decoded : ")
print(decode(encoded))
############################

# step2.py

import torch

data = torch.tensor(encode(text), dtype=torch.long)

print("First 20 token IDs : ")

print(data[:20])

print("Decoded : ")

print(decode(data[:20].tolist()))

block_size = 8

chunk = data[: block_size + 1]

print("Chunk : ", chunk)

print("Chunk decoded : ", decode(chunk.tolist()))

x = chunk[:-1]

y = chunk[1:]

print("Input x : ", x)

print("Target y : ", y)

print(decode(y.tolist()))
