
import sys
import io
import logging
import dill as pickle
from collections import Counter
import argparse
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from transformer.Constants import *



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def buildFromFile (file_path, tokenizer_type = None):
	tokenizer = get_tokenizer(tokenizer_type)

	# build vocab
	counter = Counter()
	with io.open(file_path, 'rt') as f:
		for line in f:
			counter.update(tokenizer(line))
	vocab = Vocab(counter, specials=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])
	#print('vocab:', vocab)

	examples = []
	raw_iter = iter(io.open(file_path, 'rt'))
	for raw in raw_iter:
		sentence = tokenizer(raw)
		sentence.insert(0, BOS_WORD)
		sentence.append(EOS_WORD)
		tensor = torch.tensor([vocab[token] for token in sentence], dtype=torch.long)
		examples.append(tensor)

	return vocab, examples


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input text file')
	parser.add_argument('target', type=str, help='output package file')
	opt = parser.parse_args()

	logging.info('Building package from source: %s', opt.source)
	vocab, examples = buildFromFile(opt.source)
	#print('examples:', examples)

	data = {
		'vocab': vocab,
		'examples': examples,
	}
	pickle.dump(data, open(opt.target, 'wb'))

	logging.info('Done.')



if __name__ == '__main__':
	main()
