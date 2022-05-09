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
 
 
 


# How our model is constructed #
## train() ##
The hyperparameters have been set to the best configurations that we got 

- Firstly the model taining is complete we find the validation accuracy and report the test accuracy 
- We then create a folder prediction_attention, where we create two files  `success.txt` and `failure.txt`.
- These files contain predicted words of the successful and failed predictions made by the sequence to sequence with attention.
- We randomly choose 10 inputs from our test data.
- Then we generated a 3*3 attention heatmaps on the same.

## How to use wandb for generating plots
- To use wandb: wandb.agent("<sweep_id>",train) 
- To train without wandb: train(use_wandb=False)


## validate() ##
`validate(path_to_file,folder_name)`: Finds the accuracy of the model on the test and validation dataset. 

It creates a folder ***prediction_attention*** with a sub-folder folder_name where we create two files  `success.txt` and `failure.txt`. These files contain `<input word><space><target word><space><predicted word>` of the successful and failed predictions made by the sequence to sequence to sequence.

| Parameters   | Description                                                                                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| path_to_file | Accepts parameters of type string. Contains the path to validation or test dataset. from from the folder dakshina_dataset_v1.0/hi/lexicons/                        |
| folder_name  | Accepts parameters of type string. This parameter is helpful in creating subfolder inside the prediction_attention folder as per the hyperparamter configurations. |

## visualize() ##
`visualize()`: Helps to visualise what the sequence to sequence model learns with the help of attention network.
| Parameters  | Description                                                                                                                                                                                                |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word  | Accepts string as input. Here we pass the transliterated roman word                                                                                                                                        |
| output_word | Accepts string as input. Here we pass the predicted output word.                                                                                                                                           |
| att_w       | Takes a list of list where each sublist denotes the attention weights learnt at a particular timestep. Each of the sublist of size equal to length of the transliterated roman word which is fed as input. |

**Example:**</br>
![Output_without_heatmap](https://user-images.githubusercontent.com/37553488/119277211-e53cee00-bc3b-11eb-9309-1fcf59ae18d0.png)

## connectivity() ##
`connectivity()`: Helps to visualise what the sequence to sequence model learns with the help of attention network.

| Parameters | Description                                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word | Accepts string as input. Here we pass the transliterated roman word                                                                                               |
| rnn_type   | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.                                             |
| file_path  | Accepts a string as input. Here we pass the file location where we want to store the connectivity visualisation. The visualisation is stored with the file name: [connectivity.html](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/blob/master/cs6910_assignment3/seq2seq_with_attention/connectivity.html)   |

## create_file() ##
`create_file()` is used to create and store the connectivity.html file in the specified  location.

| Parameters  | Description                                                                                                                                                       |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word  | Accepts string as input. Here we pass the transliterated roman word                                                                                               |
| text_colors | List of list where each sublist denotes the color to be given to every input character on mouse hover action on an output character.                              |
| rnn_type    | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.                                             |
| file_path   | Accepts a string as input. Here we pass the file location where we want to store the connectivity visualisation. The visualisation is stored with the file name: [connectivity.html](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/blob/master/cs6910_assignment3/seq2seq_with_attention/connectivity.html) |

## get_shade_color() ##
`get_shade_color()`: Returns a specific colour depending the value passed to it.  

| Parameters | Description                                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| value      | Expected a floating number between 0-1 denoting the attention weight of the j<sup>th</sup> input character while predicting the i<sup>th</sup>output character.  |

## transliterate() ##
`transliterate()`: Finds the predicted target word for a given tansliterated roman word, plots the attention heatmap and visualises the LSTM activations if the visual_flag is set to True.
| Parameters  | Description                                                                                                                                                                                                                                                                     |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word  | Accepts string as input. Here we pass the transliterated roman word                                                                                                                                                                                                             |
| rnn_type    | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.                                                                                                                                                           |
| file_path   | Accepts a string as input. Here we pass the file location where we want to store the attention heatmap for the input word and the predicted target word. If not specified the attention heatmaps is stored in the current working directory by the name "attention_heatmap.png" |
| visual_flag | Accepts a boolean True or boolean False. If the visual_flag is set to true then the code to statically visualise the LSTM activations are called. Default value of the flag is set to "True".                                                                                   |


## generate_inputs() ##
`generate_inputs()`: Randomly chooses 10 inputs from the test dataset and calles the transliteration() to produce the predicted target input and heatmaps. It also set the visual_flag in transliteration() to True only for the first test input and False for the rest 9 test inputs.</br>
| Parameters     | Description                                                                                                                        |
|----------------|------------------------------------------------------------------------------------------------------------------------------------|
| rnn_type       | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.              |
| n_test_samples | Accepts an integer as input. Here we pass number of test inputs to be used for the heatmap generation. Default value is set to 10. |


## plot_attention() ##
`plot_attention()`: Responsible for the attention heatmap creation.
| Parameters     | Description                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------|
| attention      | Accepts a list of attention weights learnt during the model inference for the test input.         |
| input_word     | Accepts string as input. Here we pass the input word (English word).                              |
| predicted_word | Accepts a string as input. Here we pass the prediction made by inference model.                   |
| file_name      | Accepts a string as input. Here we pass the path of the file where we want to store the heatmap.  |

Below is a sample heatmap generated by the code:</br>
![nazarandaaz](https://user-images.githubusercontent.com/37553488/119312982-a1280880-bc90-11eb-8297-a81067faeab4.png)


## How to debug the code? ##

Reduce the number of word pairs being generated by create_dataset() to some small value say 100. However reduce the batch size to a value less than 100 else you may encounter [StopIteration() error](https://stackoverflow.com/questions/48709839/stopiteration-generator-output-nextoutput-generator).

Also, to reduce the validation datasize reduce the number of inputs in validate() to some small number like 10 instead of len(input_words). 

Run the train() for 1 epoch. To visualise the code for 20 epochs you can restore the model parameters as provided in the [github repository](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/blob/master/cs6910_assignment3/seq2seq_with_attention/training_checkpoints.zip) using `checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))` after runnning the train function for one epoch.

