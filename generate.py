''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.legacy.data import Dataset
from transformer.Models import TransformerGen
from transformer.Generator import Generator


def load_model(opt, device):

	checkpoint = torch.load(opt.model, map_location=device)
	model_opt = checkpoint['settings']

	model = TransformerGen(
		model_opt.src_vocab_size,
		model_opt.trg_vocab_size,

		model_opt.src_pad_idx,
		model_opt.trg_pad_idx,

		trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
		emb_src_trg_weight_sharing=model_opt.embs_share_weight,
		d_k=model_opt.d_k,
		d_v=model_opt.d_v,
		d_model=model_opt.d_model,
		d_word_vec=model_opt.d_word_vec,
		d_inner=model_opt.d_inner_hid,
		n_layers=model_opt.n_layers,
		n_head=model_opt.n_head,
		dropout=model_opt.dropout).to(device)

	model.load_state_dict(checkpoint['model'])
	print('[Info] Trained model state loaded.')
	return model 


def main():
	'''Main Function'''

	parser = argparse.ArgumentParser(description='translate.py')

	parser.add_argument('-model', required=True,
						help='Path to model weight file')
	parser.add_argument('-data_pkl', required=True,
						help='Pickle file with both instances and vocabulary.')
	parser.add_argument('-output', default='pred.txt',
						help="""Path to output the predictions (each line will
						be the decoded sequence""")
	parser.add_argument('-beam_size', type=int, default=5)
	parser.add_argument('-max_seq_len', type=int, default=100)
	parser.add_argument('-no_cuda', action='store_true')
	parser.add_argument('-count', type=int, default=100)
	parser.add_argument('-temperature', type=float, default=1)

	opt = parser.parse_args()
	opt.cuda = not opt.no_cuda

	'''#data = pickle.load(open(opt.data_pkl, 'rb'))
	data = pickle.load(open('m30k_deen_shr.pkl', 'rb'))
	SRC, TRG = data['vocab']['src'], data['vocab']['trg']
	opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
	opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
	opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
	opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

	test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})'''

	data = pickle.load(open(opt.data_pkl, 'rb'))
	TRG = data['vocab']['trg']

	device = torch.device('cuda' if opt.cuda else 'cpu')
	generator = Generator(
		model=load_model(opt, device),
		beam_size=opt.beam_size,
		max_seq_len=opt.max_seq_len,
		#src_pad_idx=opt.src_pad_idx,
		#trg_pad_idx=opt.trg_pad_idx,
		trg_bos_idx=2,
		trg_eos_idx=3,
	).to(device)

	#unk_idx = SRC.vocab.stoi[SRC.unk_token]
	with open(opt.output, 'w') as f:
		'''for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
			#print(' '.join(example.src))
			src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
			pred_seq = generator.translate_sentence(torch.LongTensor([src_seq]).to(device))
			pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
			pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
			#print(pred_line)
			f.write(pred_line.strip() + '\n')'''
		for example in tqdm(range(opt.count), mininterval=2, desc='  - (Test)', leave=False):
			pred_seq = generator.generate_setence(temperature = opt.temperature)
			pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
			print('pred_line:', pred_line)

			f.write(pred_line.strip() + '\n')

	print('[Info] Finished.')



if __name__ == "__main__":
	'''
	Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
	'''
	main()
