import torch
import torch.optim as optim
import torch.nn as nn
from torchtext.data import Field, BucketIterator, TabularDataset
import sentencepiece as spm
import random
import math

from model2 import Encoder, Decoder, Seq2Seq

sp = spm.SentencePieceProcessor()
sp.Load('model/ja.wiki.bpe.vs5000.model')

def train(model, train_it, optimizer, criterion, clip, epoch):
    model.train()
    train_loss = 0
    for index, batch in enumerate(train_it, 1):
        src = batch.jp
        trg = batch.en
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output[1:].view(-1, output.shape[-1]), trg[1:].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        print('\repoch %03d batch %03d train_loss %6.3f' % (epoch, index, train_loss / index), end='')
    return train_loss / len(train_it)

def test(model, data_it, criterion):
    model.eval()
    test_loss = 0
    for index, batch in enumerate(data_it, 1):
        src = batch.jp
        trg = batch.en
        output = model(src, trg, 0)
        loss = criterion(output[1:].view(-1, output.shape[-1]), trg[1:].view(-1))
        test_loss += loss.item()
    print(' test_loss %6.3f' % (test_loss / index), end='')
    return test_loss / len(data_it)

def tokenize_jp(x):
    x = str(x).lower()
    # if want to reverse the order of the sentences
    # return sp.EncodeAsPieces(x)[::-1]
    return sp.EncodeAsPieces(x)

def tokenize_en(x):
    x = str(x).lower()
    x = x.translate({ord(c): None for c in '!.?,'})
    return x.split()

def translate_sentence(sentence, SRC, TRG, model, device):
    tokenized = tokenize_jp(sentence)
    numericalised = [SRC.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(numericalised).unsqueeze(1).to(device)
    translation_tensor_probs = model(tensor, None, 0).squeeze(1)
    translation_tensor = torch.argmax(translation_tensor_probs, 1)
    translation = [TRG.vocab.itos[t] for t in translation_tensor][1:]
    return translation

def main():
    SRC = Field(tokenize=tokenize_jp, init_token='<sos>', eos_token='<eos>')
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

    dataset = TabularDataset(path='data/csvfile.csv', format='csv', fields=[('id', None),('jp', SRC), ('en', TRG), ('cn', None)], skip_header=True)
    train_dt, valid_dt, test_dt = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())

    #print (len(train.examples), len(valid.examples), len(test.examples))
    #print (vars(train.examples[0]))

    SRC.build_vocab(train_dt, min_freq=2)
    TRG.build_vocab(train_dt, min_freq=2)
    #print (len(SRC.vocab), len(TRG.vocab))
    #print (SRC.vocab.freqs.most_common(10))
    #print (TRG.vocab.freqs.most_common(10))
    #print ('index 3 in the trg:', TRG.vocab.itos[3])
    #print ('index the in the trg:', TRG.vocab.stoi['the'])

    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_it, valid_it, test_it = BucketIterator.splits(
            (train_dt, valid_dt, test_dt), 
            batch_size=batch_size, 
            sort_key=lambda x: len(x.jp), 
            sort_within_batch=False, 
            device=device)

    input_dim = len(SRC.vocab)
    out_dim = len(TRG.vocab)
    enc_emb_dim = 128
    dec_emb_dim = 128
    hidden_dim = 256
    nlayers = 2
    enc_dropout = 0.3
    dec_dropout = 0.3

    enc = Encoder(input_dim, enc_emb_dim, hidden_dim, nlayers, enc_dropout)
    dec = Decoder(out_dim, dec_emb_dim, hidden_dim, nlayers, dec_dropout)

    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    pad_idx = TRG.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epochs = 30
    clip = 1
    model_save_path = 'model/s2smodel.pt'
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train(model, train_it, optimizer, criterion, clip, epoch)
        valid_loss = test(model, valid_it, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
        
        print(' train_ppl %7.3f test_ppl %7.3f' % \
                (math.exp(train_loss), math.exp(valid_loss)))

        candidate = ' '.join(vars(valid_dt.examples[2])['jp'])
        print(translate_sentence(candidate, SRC, TRG, model, device))

    model.load_state_dict(torch.load(model_save_path))
    test_loss = test(model, test_it, criterion)

    print(f' test_loss {test_loss: .3f} test_ppl {math.exp(test_loss):7.3f}')

    candidate = ' '.join(vars(valid_dt.examples[2])['jp'])
    candidate_translation = ' '.join(vars(valid_dt.examples[2])['en'])

    print(candidate)
    print(candidate_translation)
    print(translate_sentence(candidate, SRC, TRG, model, device))

    tokenized = tokenize_jp(candidate)
    numericalised = [SRC.vocab.stoi[t] for t in tokenized]
    back_to_candidate = [SRC.vocab.itos[n] for n in numericalised][1:]

    print(back_to_candidate)

if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main()
