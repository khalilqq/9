import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag
from nltk.util import ngrams
from collections import Counter

# Download needed NLTK data (only once)
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

# Helper functions
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def tokenize_text(text):
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)

def stem_tokens(tokens):
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def named_entities(text):
    tokens = tokenize_text(text)
    tags = pos_tag(tokens)
    tree = ne_chunk(tags, binary=True)
    named_entities = []
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NE':
            named_entities.append(' '.join(c[0] for c in subtree))
    return named_entities

def top_n_grams(tokens, n=3, top_k=10):
    n_grams = ngrams(tokens, n)
    return Counter(n_grams).most_common(top_k)

def process_text(file_path):
    text = load_text(file_path)
    tokens = tokenize_text(text)
    
    stems = stem_tokens(tokens)
    lemmas = lemmatize_tokens(tokens)
    
    freq_stems = FreqDist(stems).most_common(20)
    freq_lemmas = FreqDist(lemmas).most_common(20)
    
    entities = named_entities(text)
    
    return {
        'tokens': tokens,
        'stems': stems,
        'lemmas': lemmas,
        'freq_stems': freq_stems,
        'freq_lemmas': freq_lemmas,
        'named_entities': entities,
        'num_named_entities': len(entities)
    }

# Your actual file paths
files = {
    "Text 1 (Martin)": "/Users/khalilqq/Desktop/untitled folder/Martin.txt",
    "Text 2 (Lovecraft)": "/Users/khalilqq/Desktop/untitled folder/RJ_Lovecraft.txt",
    "Text 3 (Martin 2)": "/Users/khalilqq/Desktop/untitled folder/RJ_Martin.txt",
    "Text 4 (Tolkien)": "/Users/khalilqq/Desktop/untitled folder/RJ_Tolkein.txt"
}

# Processing
results = {}
for name, path in files.items():
    results[name] = process_text(path)

# Output Results
for name, data in results.items():
    print(f"\n--- {name} ---")
    print("Top 20 Stems:", data['freq_stems'])
    print("Top 20 Lemmas:", data['freq_lemmas'])
    print("Number of Named Entities:", data['num_named_entities'])

# N-gram Analysis
print("\n\n--- N-gram Analysis ---")
for name, data in results.items():
    top_ngrams = top_n_grams(data['tokens'])
    print(f"\nTop 10 3-grams for {name}:")
    for ngram, freq in top_ngrams:
        print(f"{ngram}: {freq}")
