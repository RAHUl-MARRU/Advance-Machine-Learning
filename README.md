# Advance-Machine-Learning
Deep Learning Techniques

# Predict the Next Word in Sentence using Neural Networks


# Abstract 


The aim of the project is to predict the next word in a sentence; it is also known as "language modeling." It plays a major role in NLP applications like text summarization and the translation of questions and answers into various languages. In general, a language model will take as input a raw text corpus, and the probabilities of each word in the data will be calculated to find the similarity distance between them. This text corpus is split into two portions: 70% of training data and 30% of test data. For this problem, we are now solving it by using neural networks for language modeling. The three techniques, which are RNN (recurrent neural network), LSTM (long short-term memory), and Bi-LSTM (bi-directional long short-term memory), are combined with training data. It can be trained for up to 20 epochs and evaluated with test data to determine accuracies for models. Finally, using test cases as examples, it predicts the next word of the  sentence. Note: The next word is either a word or the continuation letters of a word.

# Experiment Design 

 Following steps to implementing:- 
 
 
 • Load required packages and data from corpus.
 
 • Feature Engineering.
 
 • Building the model.
 
 • Training, Evaluating and Testing the model. 
 
 • Predicting the next word for a sentence using best model.
 
 Using Packages Keras, Tensorflow, Model – sequential Simple RNN, LSTM, Bidirectional, Dense. Activation layers, Text Feature engineering with text data is converting the string into numerical values. Applying train-test split and implementing the 120 model with training data of 70%. Finding accuracy and loss function with remaining test data of 30%.
 
 # Data
 
 The data corpus considered for this problem was collected from an English-language eBook about Project Gutenberg’s "The Adventures of Sherlock Holmes," written by author Arthur Conan Doyle, which contained various sub-contents like stories, documentation, and text data, which are necessary for our problem statement. It contains 581,884 words and 73 characters in the data corpus. In addition, we can use a variety of articles as text data 106 for this problem.
 
 # Methods
 
 The three deep learning techniques used to solve this problem are RNN (Recurrent neural network), LSTM (Long short-term memory) and Bi-LSTM (Bidirectional Long short-term memory).
 
 # Result
 
 For all the observations of three neural network models. The LSTM model has better performance than the other RNN and Bi-LSTM models. Among these RNN, got low performance standards, but Bi- LSTM is nearly identical to the LSTM algorithm.  The main criteria for considering LSTM are that it is consistent with its loss and accuracy through each epoch. Whereas in Bi-LSTM, it shows various variations in loss and accuracy scores during each epoch.
 
 Accuracy score of LSTM Model the same approach that was mentioned earlier for the RNN model also applies to the LSTM model. The performance of a model is evaluated by remaining 30% of the test data to predict the outcome with an LSTM model. Then compare the actual outcome with the predicted outcome to calculate the accuracy.
 
 ![image](https://user-images.githubusercontent.com/125625532/223234929-3e131bb2-2f10-4167-b085-674d0ed9fe81.png)
 
 
![image](https://user-images.githubusercontent.com/125625532/223235102-c5c13a7f-637d-4596-808c-56c44b60bb96.png)

#  Test Cases


Predicting the next word for the given input sentence using the best model of the three techniques  is LSTM, which has better accuracy and loss score.

![image](https://user-images.githubusercontent.com/125625532/223235471-23500275-b069-4eef-9333-ddf3233aa59a.png)


# Conclusion

To summarize, LSTM is the model with the highest accuracy and consistency. The main challenges are that running the LSTM model takes more time;However, if we train more, we can improve performance, and also if we increase the number of layers in these neural network models. We did a train-test split before applying models, and Bi-LSTM is new to our project, so it is almost as accurate or close to it when compared to previous baseline work.


