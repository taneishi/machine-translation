import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, nlayers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, nlayers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, nlayers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, nlayers, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim, 'hidden dimensions of encoder and decoder must be equal!'
        assert encoder.nlayers == decoder.nlayers, 'nlayers of the encoder and decoder must be equal!'

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
    	#inference
        if trg is None:
            trg = torch.zeros((25, src.shape[1])).fill_(2).long().to(src.device)
            assert teacher_forcing_ratio == 0, 'must be zero during inference'
            # because this is for real world usage so should not use the original target data for TRG

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device) # tensor to store decoder outputs
        hidden, cell = self.encoder(src)
        input = trg[0,:] # first input to the decoder is the <sos> token which is the first row in trg
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1] # the prediction of the next output
            input = (trg[t] if teacher_force else top1) # the next input for decoder is either the real y or the prediction
        return outputs
