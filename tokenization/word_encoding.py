import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')


# Read file
with open(
	'D:/python/keep going/tokenization/data.txt','r',encoding='utf-8') as file:
	text = file.read()
# Tokenization
def tokenize_text(text: str):
	return word_tokenize(
	text.lower()
	)
# Tokenize data
words = tokenize_text(text)

# Create vocabulary
tokens = sorted(
	list(set(words))
)

# Token -> Index
token_to_index = {
	token: index
	for index, token in enumerate(tokens)

}


# Encode
def encode(text: str):

	text_tokens = tokenize_text(text)

	indices = []

	for token in text_tokens:

		if token in token_to_index:

			indices.append(
                token_to_index[token]
			)

		else:
			print(
				f"Unknown token: {token}"
				)

	return indices


print("Vocabulary:")
print(tokens)

print()

print("Token to Index:")
print(token_to_index)

print()

print("Encode:")
print(
    encode("sau chia")
)