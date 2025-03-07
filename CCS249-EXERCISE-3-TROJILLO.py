#Cyril Reynold S. Trojillo
#BSCS 3-B

import wikipedia
import re
from collections import defaultdict

# Fetch Wikipedia Text
page = wikipedia.page("Mobile_Suit_Gundam")  # Fetch custom Wikipedia page
text = page.content[:1000]  # Limit to 1000


def preprocess_text(text):
    """Tokenizes text manually by lowercasing and removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Split by spaces
    return tokens

# Create model 
def create_bigrams(tokens):
    """Creates a list of bigrams from the tokenized text."""
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

def create_trigrams(tokens):
    """Creates a list of trigrams from the tokenized text."""
    return [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]

def train_bigram_model(tokens):
    """Trains a bigram model by counting occurrences and calculating probabilities."""
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    # Count unigrams and bigrams
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        bigram_counts[(w1, w2)] += 1
        unigram_counts[w1] += 1

    # Compute probabilities
    bigram_probs = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}
    return bigram_probs

def train_trigram_model(tokens):
    """Trains a trigram model by counting occurrences and calculating probabilities."""
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)

    # Count bigrams and trigrams
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        trigram_counts[(w1, w2, w3)] += 1
        bigram_counts[(w1, w2)] += 1

    # Compute probabilities
    trigram_probs = {trigram: count / bigram_counts[(trigram[0], trigram[1])] for trigram, count in trigram_counts.items()}
    return trigram_probs

# Perplexity Calculation
def get_bigram_probability(w1, w2, bigram_probs, smoothing=1e-6):
    """Returns probability of a bigram, using smoothing if unseen."""
    return bigram_probs.get((w1, w2), smoothing)

def get_trigram_probability(w1, w2, w3, trigram_probs, smoothing=1e-6):
    """Returns probability of a trigram, using smoothing if unseen."""
    return trigram_probs.get((w1, w2, w3), smoothing)

def calculate_perplexity(test_sentence, bigram_probs):
    """Calculates perplexity for a given test sentence."""
    test_tokens = preprocess_text(test_sentence)
    bigrams = create_bigrams(test_tokens)
    
    prob_product = 1.0
    N = len(bigrams)
    
    for w1, w2 in bigrams:
        prob_product *= get_bigram_probability(w1, w2, bigram_probs)

    perplexity = prob_product ** (-1 / N) if N > 0 else float('inf')
    return perplexity

def calculate_trigram_perplexity(test_sentence, trigram_probs):
    """Calculates perplexity for a given test sentence using trigram model."""
    test_tokens = preprocess_text(test_sentence)
    trigrams = create_trigrams(test_tokens)
    
    prob_product = 1.0
    N = len(trigrams)
    
    for w1, w2, w3 in trigrams:
        prob_product *= get_trigram_probability(w1, w2, w3, trigram_probs)

    perplexity = prob_product ** (-1 / N) if N > 0 else float('inf')
    return perplexity


# Run Everything
tokens = preprocess_text(text)  # Tokenize text
bigram_probability = train_bigram_model(tokens)
trigram_probability = train_trigram_model(tokens)

test_sentence = "Mobile Suit Gundam, also retrospectively known as First Gundam, Gundam 0079 or simply Gundam '79, is a Japanese anime television series produced Nippon Sunrise."
bigram_perplexity = calculate_perplexity(test_sentence, bigram_probability)
trigram_perplexity = calculate_trigram_perplexity(test_sentence, trigram_probability)

# Print result
print(f"Bigram model perplexity: \"{test_sentence}\" = Score: {bigram_perplexity}")
print(f"Trigram model perplexity: \"{test_sentence}\" = Score: {trigram_perplexity}")