# Assignment 3
## Using RNN with attention mechanism to build a transliteration system
- In this assignment we had implemented attention networks to overcome the limitations of vanilla seq2seq model 
- We had also visualised the interactions between different components in an RNN based model. 
- We used wandb for hyper parameter configuration using the validation dataset and visualisation of test data. 
- Inferences had been made after making large number of runs.
### Note:
We had used a single layered encoder and a single layered decoder
We had added an attention network to this basic seq_to_seq and trained the model.

## Imported Libraries 
- matplotlib libraries for necessary plots like heatmap
- Keras and tensorflow
- os for file operations
- io for file operations
- time to measure the time taken for every epoch.
- shutil for deleting a folder forcibly
- Ipython for visualisation
-  numpy to convert the attention weights and loss values from tensor objects into numpy objects.
- wandb to generate plot and make report
- random to pick 10 randomly selected heatplots

The Nirmala.ttf file is used for Hindi font file for displaying Hindi characters in the matplotlib plots. Before generating the heat map, we need to upload the same.
Uncomment wandb.agent() and comment the call to train()
An additional thing to do is set use_wandb=False while calling train function if you do not desire to log results into wandb
To execute the program in Google Colab one can click on the run cell option or download the notebook and run.

## How program is written
At first the program downloads the dataset 

Then it sets the train, validation and test filepath

Then it creatres Input word tensors, target word tensors, input vocabulary and output vocabulary from training set

Every target word and input word is appended with a Start of Sequence Character '\t' and End of Sequence Character '\n'.

Then training begins.

Then test accuracy is reported.

Random function is used to generate random of test inputs to evaluate the model performance.

The target word for each of the test inputs had been predicted.

Followed by this attention heatmap is generated.

RNN Type activations are visualized statistically.

Perform connectivity visualisation to understand which is the input(ENGLISH) character that the sequence to sequence model is looking at

This is done while decoding the ith character in the output(a HINDI character)

## Acknowledgements 
1.	Youtube playlist of Associate Proffessor Mitesh Khapra, Indian Institute of Technology, Madras - https://www.youtube.com/watch?v=aPfkYu_qiF4&list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU
2.	Blog in Analytics Vidhya for RNN - https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/Blog on dropout for regularizing – 
3.	Blog for Dropout – https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
4.	Additional knowledge on basics of CNN - https://datascience.stackexchange.com/questions/64022/
5.	Pre-training vs Fine tuning -- https://stackoverflow.com/questions/68461204/continual-pre-training-vs-fine-tuning-a-language-model-with-mlm 
6.	Training in Keras –
https://stackoverflow.com/questions/52031587/how-can-i-make-a-trainable-parameter-in-keras
7.	Guide to RNN Blog –
 https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9

