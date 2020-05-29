# Machine translation using LSTM-based encoder-decoder model in PyTorch

## Model

This is the implementation of

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

adapted to JP-EN translation. These models are also known as seq2seq models.
  
## Data

- https://nlp.stanford.edu/projects/jesc/, official split.

- [JEC_basic_sentence_v1-2.xls](http://nlp.ist.i.kyoto-u.ac.jp/EN/), converted into csv with pandas (preprocess.py).

Japanese is tokenized using [sentencepiece](https://github.com/google/sentencepiece/), English is tokenized using space (sorry, too lazy).

- [SentencePiece model](https://nlp.h-its.org/bpemb/ja/ja.wiki.bpe.vs5000.model)
