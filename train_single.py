'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy.data import Dataset, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import TransformerGen
from transformer.Optim import ScheduledOptim



def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
	''' Apply label smoothing if needed '''

	loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

	pred = pred.max(1)[1]
	gold = gold.contiguous().view(-1)
	non_pad_mask = gold.ne(trg_pad_idx)
	n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
	n_word = non_pad_mask.sum().item()

	return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
	''' Calculate cross entropy loss, apply label smoothing if needed. '''

	gold = gold.contiguous().view(-1)

	if smoothing:
		eps = 0.1
		n_class = pred.size(1)

		one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
		one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
		log_prb = F.log_softmax(pred, dim=1)

		non_pad_mask = gold.ne(trg_pad_idx)
		loss = -(one_hot * log_prb).sum(dim=1)
		loss = loss.masked_select(non_pad_mask).sum()  # average later
	else:
		loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
	return loss


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
	''' Epoch operation in training phase'''

	model.train()
	total_loss, n_word_total, n_word_correct = 0, 0, 0 

	desc = '  - (Training)   '
	for batch in tqdm(training_data, mininterval=1, desc=desc, leave=False):

		# prepare data
		trg_seq, gold = batch[:, :-1], batch[:, 1:].contiguous().view(-1)

		# forward
		optimizer.zero_grad()
		pred = model(None, trg_seq)

		# backward and update parameters
		loss, n_correct, n_word = cal_performance(
			pred, gold, opt.pad_idx, smoothing=smoothing) 
		loss.backward()
		optimizer.step_and_update_lr()

		# note keeping
		n_word_total += n_word
		n_word_correct += n_correct
		total_loss += loss.item()

	loss_per_word = total_loss/n_word_total
	accuracy = n_word_correct/n_word_total
	return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
	''' Epoch operation in evaluation phase '''

	model.eval()
	total_loss, n_word_total, n_word_correct = 0, 0, 0

	desc = '  - (Validation) '
	with torch.no_grad():
		for batch in tqdm(validation_data, mininterval=1, desc=desc, leave=False):

			# prepare data
			trg_seq, gold = batch[:, :-1], batch[:, 1:].contiguous().view(-1)

			# forward
			pred = model(None, trg_seq)
			loss, n_correct, n_word = cal_performance(
				pred, gold, opt.pad_idx, smoothing=False)

			# note keeping
			n_word_total += n_word
			n_word_correct += n_correct
			total_loss += loss.item()

	loss_per_word = total_loss/n_word_total
	accuracy = n_word_correct/n_word_total
	return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
	''' Start training '''

	# Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
	if opt.use_tb:
		print("[Info] Use Tensorboard")

		import tensorflow as tf
		import tensorboard.compat.tensorflow_stub as stub
		tf.io.gfile = stub.io.gfile

		from torch.utils.tensorboard import SummaryWriter
		tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

	log_train_file = os.path.join(opt.output_dir, 'train.log')
	log_valid_file = os.path.join(opt.output_dir, 'valid.log')

	print('[Info] Training performance will be written to file: {} and {}'.format(
		log_train_file, log_valid_file))

	with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
		log_tf.write('epoch,loss,ppl,accuracy\n')
		log_vf.write('epoch,loss,ppl,accuracy\n')

	def print_performances(header, ppl, accu, start_time, lr):
		print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.4f} %, lr: {lr:.4e}, '\
			  'elapse: {elapse:3.3f} min'.format(
				  header=f"({header})", ppl=ppl,
				  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

	#valid_accus = []
	valid_losses = []
	for epoch_i in range(opt.epoch):
		print('[ Epoch', epoch_i, ']')

		start = time.time()
		train_loss, train_accu = train_epoch(
			model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
		train_ppl = math.exp(min(train_loss, 100))
		# Current learning rate
		lr = optimizer._optimizer.param_groups[0]['lr']
		print_performances('Training', train_ppl, train_accu, start, lr)

		start = time.time()
		valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
		valid_ppl = math.exp(min(valid_loss, 100))
		print_performances('Validation', valid_ppl, valid_accu, start, lr)

		valid_losses += [valid_loss]

		checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

		if opt.save_mode == 'all':
			model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
			torch.save(checkpoint, model_name)
		elif opt.save_mode == 'best':
			model_name = 'model.chkpt'
			if valid_loss <= min(valid_losses):
				torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
				print('	- [Info] The checkpoint file has been updated.')

		with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
			log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
				epoch=epoch_i, loss=train_loss,
				ppl=train_ppl, accu=100*train_accu))
			log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
				epoch=epoch_i, loss=valid_loss,
				ppl=valid_ppl, accu=100*valid_accu))

		if opt.use_tb:
			tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
			tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
			tb_writer.add_scalar('learning_rate', lr, epoch_i)

def main():
	''' 
	Usage:
	python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
	'''

	parser = argparse.ArgumentParser()

	parser.add_argument('-data_pkl', default=None)	 # all-in-1 data pickle or bpe field

	parser.add_argument('-epoch', type=int, default=10)
	parser.add_argument('-b', '--batch_size', type=int, default=2048)

	parser.add_argument('-d_model', type=int, default=512)
	parser.add_argument('-d_inner_hid', type=int, default=2048)
	parser.add_argument('-d_k', type=int, default=64)
	parser.add_argument('-d_v', type=int, default=64)

	parser.add_argument('-n_head', type=int, default=8)
	parser.add_argument('-n_layers', type=int, default=6)
	parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
	parser.add_argument('-lr_mul', type=float, default=2.0)
	parser.add_argument('-seed', type=int, default=None)
	parser.add_argument('-n_seq_max_len', type=int, default=0x400)

	parser.add_argument('-dropout', type=float, default=0.1)
	#parser.add_argument('-embs_share_weight', action='store_true')
	parser.add_argument('-proj_share_weight', action='store_true')
	parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

	parser.add_argument('-output_dir', type=str, default=None)
	parser.add_argument('-use_tb', action='store_true')
	parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

	parser.add_argument('-no_cuda', action='store_true')
	parser.add_argument('-label_smoothing', action='store_true')

	opt = parser.parse_args()
	opt.cuda = not opt.no_cuda
	opt.d_word_vec = opt.d_model

	# https://pytorch.org/docs/stable/notes/randomness.html
	# For reproducibility
	if opt.seed is not None:
		torch.manual_seed(opt.seed)
		torch.backends.cudnn.benchmark = False
		# torch.set_deterministic(True)
		np.random.seed(opt.seed)
		random.seed(opt.seed)

	if not opt.output_dir:
		print('No experiment result will be saved.')
		raise

	if not os.path.exists(opt.output_dir):
		os.makedirs(opt.output_dir)

	if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
		print('[Warning] The warmup steps may be not enough.\n'\
			  '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
			  'Using smaller batch w/o longer warmup may cause '\
			  'the warmup stage ends with only little data trained.')

	device = torch.device('cuda' if opt.cuda else 'cpu')

	#========= Loading Dataset =========#

	if opt.data_pkl:
		training_data, validation_data = prepare_dataloaders(opt, device)
	else:
		raise

	print(opt)

	transformer = TransformerGen(
		opt.vocab_size,
		opt.vocab_size,
		src_pad_idx=opt.pad_idx,
		trg_pad_idx=opt.pad_idx,
		trg_emb_prj_weight_sharing=opt.proj_share_weight,
		d_k=opt.d_k,
		d_v=opt.d_v,
		d_model=opt.d_model,
		d_word_vec=opt.d_word_vec,
		d_inner=opt.d_inner_hid,
		n_layers=opt.n_layers,
		n_head=opt.n_head,
		n_position=opt.n_seq_max_len,
		dropout=opt.dropout,
		scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

	optimizer = ScheduledOptim(
		optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
		opt.lr_mul, opt.d_model, opt.n_warmup_steps)

	train(transformer, training_data, validation_data, optimizer, device, opt)


def batchize (data, batch_size, n_seq_max_len, device):
	def splitSentence (sentence, max_len):
		OVERLAPPED = max_len // 3

		s_len = len(sentence)
		if s_len <= max_len:
			return [sentence]

		result = []
		for i in range(0, s_len, max_len - OVERLAPPED):
			if s_len - i < max_len:
				result.append(sentence[-max_len:])
			else:
				result.append(sentence[i:i + max_len])

		return result

	# split long sentence
	sentenceLists = [splitSentence(sentence, n_seq_max_len) for sentence in data]
	data = [sentence for sentences in sentenceLists for sentence in sentences]

	result = []
	for i in range(0, len(data), batch_size):
		#records = [torch.reshape(record, (1, -1)) for record in data[i:min(i + batch_size, len(data))]]
		records = data[i:min(i + batch_size, len(data))]
		seq_len = max([len(x) for x in records])
		fixed_records = [torch.full((1, seq_len), 1) for i in range(batch_size)]
		for fixed, var in zip(fixed_records, records):
			fixed[0, :len(var)] = var

		batch = torch.cat(fixed_records, dim=0).to(device)
		result.append(batch)

	return result


def prepare_dataloaders(opt, device):
	batch_size = opt.batch_size
	n_seq_max_len = opt.n_seq_max_len
	data = pickle.load(open(opt.data_pkl, 'rb'))

	#opt.max_token_seq_len = data['settings'].max_len
	opt.pad_idx = data['vocab'].stoi[Constants.PAD_WORD]

	opt.vocab_size = len(data['vocab'])

	examples = data['examples']
	train_examples = [sentence for i, sentence in enumerate(examples) if i % 10 in [1, 2, 3, 4, 5, 6, 7, 8]]
	valid_examples = [sentence for i, sentence in enumerate(examples) if i % 10 in [9]]

	train_examples = batchize(train_examples, batch_size, n_seq_max_len, device)
	valid_examples = batchize(valid_examples, batch_size, n_seq_max_len, device)

	return train_examples, valid_examples



if __name__ == '__main__':
	main()
