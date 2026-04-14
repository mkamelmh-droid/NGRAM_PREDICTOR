from normalizer import Normalizer
from ngram_model import NGramModel

norm = Normalizer()
training_text = norm.load('data')
norm.process_and_save('data', 'data/eval_tokens.txt')
model = NGramModel()
model.build_vocab('data/eval_tokens.txt')
model.build_counts_and_probabilities('data/eval_tokens.txt')
model.save_model('data/model.json')
model.save_vocab('data/vocab.json')