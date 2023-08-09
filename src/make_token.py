# Provides a function for tokenization.

from keras.preprocessing.text import hashing_trick
from hashlib import sha256

VOCAB_SIZE = 48997

# The total vocab size was determined by:
# 1. Loading the `content` column of the CSV into a list.
# 2. " ".join(list) to get a single string of all the words.
# 3. text_to_word_sequence on the single string
# 4. `set()`ing the list, and getting the length. (set can only contain 1 instance of each word)

def make_hash(text: str) -> int:
  hash = sha256(text.encode())
  digest = hash.hexdigest() # Convert the input text into a hexadecimal string.
  return int(digest,16) # Convert the hexadecimal string to an integer

def tokenize(text: str) -> list[int]:
  tokens = hashing_trick(text, round(VOCAB_SIZE*1.3), hash_function=make_hash)
  diff = 40-len(tokens)
  if diff > 0:
    for _ in range(diff):
      tokens.append(0)
  # because this could still end up with >40 tokens, trim it down to 40 max.
  tokens = tokens[:40]
  return tokens