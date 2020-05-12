# Machine_Translation_Seq2Seq

Machine translation (jp-en) using LSTM-based encoder-decoder model (PyTorch).
This is the implementation of several models:

- model.py, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- model2.py, [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

adapted to JP-EN translation.
  
# Data

- https://nlp.stanford.edu/projects/jesc/, official split.

- [JEC_basic_sentence_v1-2.xls](http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?JEC%20Basic%20Sentence%20Data)

The xls data is converted into csv with panda (prepro.py).

Japanese is tokenized using sentencepiece (https://github.com/google/sentencepiece/), English is tokenized using space (sorry, too lazy).

- SentencePiece model 
    - https://nlp.h-its.org/bpemb/ja/ja.wiki.bpe.vs5000.model
