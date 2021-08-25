# COMP9444 Neural Networks and Deep Learning - Japanese Character Recognition
You will be implementing networks to recognize handwritten Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or KMNIST for short. It is worth reading, 
but in short: significant changes occurred to the language when Japan reformed their education system in 1868, 
and the majority of Japanese today cannot read texts published over 150 years ago. This paper presents a dataset of handwritten, labeled examples of this 
old-style script (Kuzushiji). Along with this dataset, however, they also provide a much simpler one, containing 10 Hiragana characters with 7000 samples per 
class. This is the dataset we will be using.

1. Implement a model NetLin which computes a linear function of the pixels in the image, followed by log softmax. Run the code by typing:
> python3 kuzu_main.py --net lin

Copy the final accuracy and confusion matrix into your report. The final accuracy should be around 70%. Note that the rows of the confusion matrix indicate the 
target character, while the columns indicate the one chosen by the network. (0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha", 6="ma", 7="ya", 8="re", 9="wo").

2. Implement a fully connected 2-layer network NetFull (i.e. one hidden layer, plus the output layer), using tanh at the hidden nodes and log softmax at the 
output node. Run the code by typing:
> python3 kuzu_main.py --net full

Try different values (multiples of 10) for the number of hidden nodes and try to determine a value that achieves high accuracy (at least 84%) on the test set. 
Copy the final accuracy and confusion matrix into your report.

3. Implement a convolutional network called NetConv, with two convolutional layers plus one fully connected layer, all using relu activation function, followed by 
the output layer, using log softmax. You are free to choose for yourself the number and size of the filters, metaparameter values (learning rate and momentum), 
and whether to use max pooling or a fully convolutional architecture. Run the code by typing:
> python3 kuzu_main.py --net conv

Your network should consistently achieve at least 93% accuracy on the test set after 10 training epochs. Copy the final accuracy and confusion matrix into your 
report.

4. Briefly discuss the following points:
- the relative accuracy of the three models,
- the confusion matrix for each model: which characters are most likely to be mistaken for which other characters, and why?
