import argparse
import numpy as np
import torch
import torch.nn as nn
import model
import data
from match_word2vecs import load_embeddings, getWordvectors
import pickle

layernames = ['AbsoluteValue','Concat','Convolution2D','Dense','Dropout','Eltwise','Flatten','LRN','BatchNormalization','Pooling2D','Power','ReLU','Sigmoid','Tanh']
Attributes = ['nb_filter','kernel_row','stride_col','border_mode','init','bias','output_dim','probability','operation','local_size']
noparamlayers = ['AbsoluteValue','Eltwise','Flatten','BatchNormalization','Power','ReLU','Sigmoid','Tanh']
model_root_path = '../models/'
rnn_model_path = model_root_path + 'model_bs512_lr20.pt'
dictionary_path = model_root_path + 'dict.pkl'
criterion = nn.CrossEntropyLoss()

def myargparser():
    parser = argparse.ArgumentParser(description='Model2Representation')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--data', type=str, default='../data/lm_model/',help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch size during evaluation')
    parser.add_argument('--bptt', type=int, default=35,help='sequence length')
    parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
    parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
    parser.add_argument('--testOnly', action='store_true', help='Perform only testing')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    return parser

def model2sentence(model_nlds):
    sentence = ''
    for key in model_nlds['nldsJson']['layers']:
        if key['layer_type'] in layernames:
            params = key['layer_params']
            if key['layer_type'] == 'Convolution2D':
                #assert(params['kernel_col']==params['kernel_row'])
                #assert(params['stride_col']==params['stride_row'])
                word = key['layer_type']+'_'+str(params['nb_filter'])+'_'+str(params['kernel_row'])+'_'+str(params['kernel_col'])+'_'+str(params['stride_row'])+'_'+str(params['stride_col'])+'_'+str(params['border_mode'])
            elif key['layer_type'] in noparamlayers:
                word = key['layer_type']
            elif key['layer_type'] == 'Pooling2D':
                word = key['layer_type']+'_'+str(params['kernel_row'])+'_'+str(params['kernel_col'])+'_'+str(params['stride_row'])+'_'+str(params['stride_col'])+'_'+str(params['function'])+'_'+str(params['border_mode'])
            elif key['layer_type'] == 'Dropout':
                word = key['layer_type']+'_'+str(params['probability'])
            elif key['layer_type'] == 'Eltwise':
                word = key['layer_type']+'_'+str(params['operation'])
            elif key['layer_type'] == 'Dense':
                word = key['layer_type']+'_'+str(params['output_dim'])+'_'+str(params['init'])
            elif key['layer_type'] == 'LRN':
                word = key['layer_type']+'_'+str(params['local_size'])
            else:
                print(key)
                print('Error')
            word = word+' '
            sentence+=word
            #print(word)
    return sentence

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# | a g m s | | b h n t |
# | b h n t | | c i o u |
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i, opt):
    seq_len = min(opt.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source, model, dictionary, opt):
    # Turn on evaluation mode which disables dropout. Gives perplexity
    model.eval()
    total_loss = 0.
    ntokens = len(dictionary)
    hidden = model.init_hidden(opt.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, opt.bptt):
            data, targets = get_batch(data_source, i, opt)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, 13580)
            total_loss += len(data) * criterion(output_flat, targets[0]).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

def test(data_source, model, dictionary, opt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(dictionary)
    hidden = model.init_hidden(opt.eval_batch_size)
    listhid = []
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, opt.bptt):
            data, targets = get_batch(data_source, i, opt)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            listhid.append(hidden[-1].cpu().numpy())
            hidden = repackage_hidden(hidden)
    return listhid

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data#.to(device)

def get_representation(model_nlds):
    parser = myargparser()
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        if not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if opt.cuda else "cpu")

    with open(rnn_model_path, 'rb') as f:
        rnn_model = torch.load(f, map_location='cpu')
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        rnn_model.rnn.flatten_parameters()

    with open(dictionary_path, 'rb') as f:
        dictionary = pickle.load(f)

    sentence = model2sentence(model_nlds)
    corpus = data.Corpus("", dictionary)
    eval_batch_size = 3
    test_data_2 = batchify(corpus.tokenize_sentence(sentence), eval_batch_size).to(device)
    test_data_2 = test_data_2.view(1, -1).t().contiguous()
    test_data_2.to(device)
    listhid = test(test_data_2, rnn_model, dictionary, opt)
    model_representation = np.squeeze(np.transpose(np.sum(listhid[0],axis=0)))
    return model_representation


if __name__ == '__main__':
    import json
    nlds_file_path = '../data/nlds/resnet_20'
    with open(nlds_file_path) as dump:
        model_nlds = json.load(dump)
    get_representation(model_nlds)
    sentence = model2sentence(model_nlds)
    print(sentence)