# Vanilla Seq-to-Seq
 Here we have implemented the Encoder-Decoder model without the attention. We have built stacked encoder-decoder model here.
# Set up and Installation: #
Easiest way to try this code out is :-
1. Download the .ipynb file
2. Upload it to Google Colab
3. Use the "Run All Cells" option to begin training the best model.

We got an accuracy of 35.00%

# Methods #
|   | Method name     | Description                                                                                                          |
|---|-----------------|----------------------------------------------------------------------------------------------------------------------|
| 1 | data            | Processing the data by padding the output and followed by tokenizing it .                                                      |
| 2 | build_model     | As required by parameters create a model without compiling it                                                        |
| 3 | build_inference | Decoding the given input                                          |
| 4 | decode_batch    | Batch of inputs is decoded using inference model                                                          |
| 5 | test_accuracy   | Gives the accuracy of the model using the test data. This generates the two text files found in Predictions_Vanilla folder  |
| 6 | batch_validate  | Gives the accuracy of the model using validation data                                                              |
| 7 | train           | Training is done using configs sent by wandb.                                                                                   |
| 8 | manual_train    | Traininig is done using our custom configurations.                                                                                      |




# Training without WandB #

As shown below manual training takes in a object of 'configuration' class.

If you want to train without generating plots in wandb Set wb=False

config=configuration('LSTM',32,512,3,2,.3,20,64)
manual_train(config)

- Argument 1 : Cell Type
- Argument 2 : Embedding Dimension
- Argument 3 : Hidden Units
- Argument 4 : Number of encoder layers
- Argument 5 : Number of decoder layers
- Argument 6 : Dropout
- Argument 7 : Epochs
- Argument 8 : Batch Size

# Training with WandB #
To generate plots, Set wb=True

Then provide your sweep ID while running the cell.

# Best Configurations
- Cell Type : LSTM
- Embedding Size : 32
- Hidden Units : 256
- Number of encoder layer : 4
- Number of decoder layer : 4
- Dropout : 0.3
- Epochs : 20
- Batch Size : 64


# Acknoledgements #
1.	Youtube playlist of Associate Proffessor Mitesh Khapra, Indian Institute of Technology, Madras - https://www.youtube.com/watch?v=aPfkYu_qiF4&list=PLEAYkSg4uSQ1r2XrJ_GBzzS6I-f8yfRU 
2.	Blog in Analytics Vidhya for RNN - https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neuralnetworks/Blog on dropout for regularizing –  
3.	Blog for Dropout – https://machinelearningmastery.com/dropout-for-regularizingdeep-neural-networks/ 
4.	Additional knowledge on basics of CNN - https://datascience.stackexchange.com/questions/64022/ 
5.	Pre-training vs Fine tuning -- https://stackoverflow.com/questions/68461204/continual-pre-training-vs-finetuning-a-language-model-with-mlm  
6.	Training in Keras – https://stackoverflow.com/questions/52031587/how-can-i-make-a-trainableparameter-in-keras 
7.	Guide to RNN Blog –  https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks79e5eb8049c9 


