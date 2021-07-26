''' Generate text with trained model. '''

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
		model_opt.vocab_size,
		model_opt.vocab_size,

		model_opt.pad_idx,
		model_opt.pad_idx,

		trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
		emb_src_trg_weight_sharing=model_opt.embs_share_weight,
		d_k=model_opt.d_k,
		d_v=model_opt.d_v,
		d_model=model_opt.d_model,
		d_word_vec=model_opt.d_word_vec,
		d_inner=model_opt.d_inner_hid,
		n_layers=model_opt.n_layers,
		n_head=model_opt.n_head,
		n_position=model_opt.n_seq_max_len,
		dropout=model_opt.dropout,
	).to(device)

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
	parser.add_argument('-max_seq_len', type=int, default=0x100)
	parser.add_argument('-no_cuda', action='store_true')
	parser.add_argument('-no_word_sep', action='store_true')
	parser.add_argument('-count', type=int, default=100)
	parser.add_argument('-temperature', type=float, default=1)

	opt = parser.parse_args()
	opt.cuda = not opt.no_cuda

	data = pickle.load(open(opt.data_pkl, 'rb'))
	vocab = data['vocab']
	vocab = vocab['trg'].vocab if hasattr(vocab, 'get') and vocab.get('trg') else vocab

	device = torch.device('cuda' if opt.cuda else 'cpu')
	generator = Generator(
		model=load_model(opt, device),
		max_seq_len=opt.max_seq_len,
		trg_bos_idx=vocab.stoi[Constants.BOS_WORD],
		trg_eos_idx=vocab.stoi[Constants.EOS_WORD],
	).to(device)

	word_sep = '' if opt.no_word_sep else ' '

	with open(opt.output, 'w') as f:
		for example in tqdm(range(opt.count), mininterval=1, desc='  - (Test)', leave=False):
			pred_seq = generator.generate_sentence(temperature = opt.temperature)
			pred_line = word_sep.join(vocab.itos[idx] for idx in pred_seq)
			#print('pred_line:', pred_line)

			f.write(pred_line.strip() + '\n')
			f.flush()

	print('[Info] Finished.')



if __name__ == "__main__":
	'''
	Usage: python generate.py -model trained.chkpt -data multi30k.pt -no_cuda
	'''
	main()
