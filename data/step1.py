#step1.py

with open("training.txt","r",encoding="utf-8") as f:
    text = f.read()

print("training text : ")
print(text)

chars = sorted(list(set(text)))

vocab_size = len(chars)

print("\n vocabulary : ")

print(chars)

print(" vocabulary size : ", vocab_size)

char_to_id = {char:index for index,char in enumerate(chars)}

id_to_char = {index: char for index,char in enumerate(chars)}

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