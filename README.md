# PermLSTM
This is a demo that using permuted diagonal mask matrices to prune the LSTM network. The original network performs speech recognition tasks on the TIMIT dataset.

Requirements:
1. This example uses pytorch and several other libaries. I recommend installing Anaconda 4.0.0 from https://repo.continuum.io/archive/ for Python 3. This saves you a lot of effort installing necessary packages yourself. The current Anaconda version is 4.4, but it seems the newest version has a problem when installing the CTC decoder later. Also, python 2.7 has a problem with the CTC decoder, which has not been clearly solved. So, to save effort, Anaconda 4.0.0 for Python 3.5 is the best starting point.
2. The code uses Pytorch. An easy installation can be found here http://pytorch.org/. You need to have GPU and CUDA.
3. Pytorch does not provide the CTC loss function. However, deepspeech has a pytorch wrapper for the CTC loss function. Follow the instructions at https://github.com/SeanNaren/deepspeech.pytorch to install the Warp-CTC module. After installation, copy libwarpctc.so to your python lib anaconda3/lib/python3.5/site-packages/
4. If you want to use the CTC beam decoder as in the paper http://proceedings.mlr.press/v32/graves14.pdf, you need to have the decoder libary installed. Follow the instructions at https://github.com/ryanleary/pytorch-ctc
Run the example:
I have provided the processed features in train_ctc and cv_ctc. To train and prune the CTC model, run the script run.py. You can specify the sparsity(defined by the variable p), LSTM layers, number of directions, etc at the beginning of this script. I assume that your features are 40-dimension. If not, you can change it in __main__. Note, the num_classes should be set to 61, not 62, because the network will automatically add 1 for the blank label. The details of the model is in set_model_ctc.py. The models will be saved in weights_ctc, in which I have generated some models already. 
