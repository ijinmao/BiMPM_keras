# BiMPM_keras
Keras implementation of Bilateral Multi-Perspective Matching [1] using in [Quora Question Duplicate Pairs Competition](https://www.kaggle.com/c/quora-question-pairs). You can find the original tensorflow implementation from [here](https://github.com/zhiguowang/BiMPM). 

## Description

`train_model.py` - train and test BiMPM model.

`bimpm.py` - model graph.

`multi_perspective.py` - multi perspective layer.

`config.py` - hyper-parameters.

`layers.py` - other layers, word embedding layers, context layer, etc.

## Requirements

- python 2.7
- tensorflow 1.1.0
- keras 2.0.3
- numpy 1.12.1
- pandas 0.19.2
- nltk 3.2.2
- gensim 1.0.1

## References

[[1]](https://arxiv.org/pdf/1702.03814) Zhiguo Wang, Wael Hamza and Radu Florian. "Bilateral Multi-Perspective Matching for Natural Language Sentences."



