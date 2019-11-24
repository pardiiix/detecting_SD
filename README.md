# detecting_SD

import nltk, re, autocorrect, pandas, numpy, keras, tensorflow, sklearn, matplotlib


First, copy and paste glove 100d to directory. Then, run CNN_1D.py. This trains the model and saves it to saved_cnn_model.h5.
After the model has been trained, run predict_model_confidence.py. This will give you the predictions + ask you to  label the comments that the network was not sure about. (open any csv file in the code that you need to examine).

--------------------------------------------------------------------------

Or, you can just use the saved_cnn_model.h5, which I have trained, and run run predict_model_confidence.py.

--------------------------------------------------------------------------
