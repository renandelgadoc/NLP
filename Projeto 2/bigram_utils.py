import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import tiktoken
import random

enc = tiktoken.get_encoding("cl100k_base")

def generate_bigram(corpus):     

	sentences = [f"<|endoftext|> {sentence} <|endoftext|>" for sentence in nltk.sent_tokenize(corpus)]
 
	probabilities = {} # probabilidade do bigrama, probabilidade de ch2 dado ch1
	sums = {}	
 
	#preenche dicionarios
	for sentence in sentences:
		tokens = enc.encode(sentence,allowed_special="all")
		for ch1, ch2 in zip(tokens, tokens[1:]):
			sums[ch1] = sums.get(ch1, 0) + 1
			# smoothing começando com 1
			probabilities[(ch1,ch2)] = probabilities.get((ch1, ch2), 1) + 1

   # calcula probabilidades
	for key in probabilities:
		probabilities[key] /= sums[key[0]]
	
	return probabilities
	
def perplexity(probabilities, test_text):
    # tokeniza o teste
	with open(test_text, 'r') as f:
		test = f.read()
	sentences = [f"<|endoftext|> {sentence} <|endoftext|>" for sentence in nltk.sent_tokenize(test)]
	tokens_test = enc.encode("".join(sentences),allowed_special="all")

	# calcula perplexidade do teste baseado nas probabilidades do treino
	perplexity = 1
	N = len(tokens_test)
	for ch1, ch2 in zip(tokens_test, tokens_test[1:]):
		perplexity = (perplexity * ((1/probabilities.get((ch1,ch2), 1))) ** (1/N))
	return perplexity
    
 
def generate_text(probablities, num_of_tokens):
	token = 100257 # token inicial
	i = 0
	text = ""
	for i in range(num_of_tokens):
		# Pega todas as chaves do dicionario que o primeiro termo da tupla = token
		keys = [key[1] for key in probablities if key[0] == token]
  
  		# Pega todas as probabilidades do dicionario que o primeiro termo da tupla = token
		values = [probablities[key] for key in probablities if key[0] == token]
  
		# escolhe um termo aleatorio
		generated_text = random.choices(keys, weights=values, k=1)
	
		# token = termo escolhido
		token = generated_text[0]
  
		text += enc.decode_single_token_bytes(token).decode("utf-8", errors="ignore")

	return text


