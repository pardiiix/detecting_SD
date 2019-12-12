# detecting_SD

import nltk, re, autocorrect, pandas, numpy, keras, tensorflow, sklearn, matplotlib, glob, os

---------------------------------------------------
Run main.py to train with the manualy labeled data (SD and NSD files). Then it predicts the unlabeled data with trained parameters and asks for labels of the data which it cannot decide whether they are self disclosing or not. It then trains all over again with the labeled data which it is almost certain of (self-learning) and the data which it was not certain at first but asked the user (active learning).
Then, run final_prediction.py to get the prediction on unlabeled data which we would like to analyse.

----------------------------------------------------


You can also run CNN_1D, CNN_2D, simpleRNN_model, GRU_model, LSTM_model and bi_LSTM_model to find the best model with the first set of labeled data.


Note: to run CNN_1D, call the function create_cnn_model()

Note: Add a glove100d text file to the root
