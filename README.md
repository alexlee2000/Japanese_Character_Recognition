# COMP9444 Neural Networks and Deep Learning - Japanese Character Recognition
You will be implementing networks to recognize handwritten Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or KMNIST for short. Significant changes occurred to the language when Japan reformed their education system in 1868, and the majority of Japanese today cannot read texts published over 150 years ago.
The dataset we will be using contains 10 Hiragana characters with 7000 samples per class.

1. Implement a model NetLin which computes a linear function of the pixels in the image, followed by log softmax. Run the code by typing:
> python3 kuzu_main.py --net lin

Produce the final accuracy and confusion matrix. Note that the rows of the confusion matrix indicate the target character, while the columns indicate the one chosen by the network. (0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha", 6="ma", 7="ya", 8="re", 9="wo").

2. Implement a fully connected 2-layer network NetFull (i.e. one hidden layer, plus the output layer), using tanh at the hidden nodes and log softmax at the 
output node. Run the code by typing:
> python3 kuzu_main.py --net full

Try different values (multiples of 10) for the number of hidden nodes and try to determine a value that achieves high accuracy (at least 84%) on the test set. 
Produce the final accuracy and confusion matrix.

3. Implement a convolutional network called NetConv, with two convolutional layers plus one fully connected layer, all using relu activation function, followed by 
the output layer, using log softmax. You are free to choose for yourself the number and size of the filters, metaparameter values (learning rate and momentum), 
and whether to use max pooling or a fully convolutional architecture. Run the code by typing:
> python3 kuzu_main.py --net conv

Your network should consistently achieve at least 93% accuracy on the test set after 10 training epochs. Produce the final accuracy and confusion matrix.

4. Briefly discuss the following points:
- the relative accuracy of the three models,
- the confusion matrix for each model: which characters are most likely to be mistaken for which other characters, and why?

## Part 1 - NetLin 
Final Accuracy
Test set: Average loss: 1.0102, Accuracy: 6967/10000 (70%)

Confusion Matrix

![Screen Shot 2021-08-27 at 1 32 10 pm](https://user-images.githubusercontent.com/43845085/131067715-55f29c36-016a-43b6-9c0f-989af14ff9b0.png)

## Part 2 - NetFull
Final Accuracy
Test set: Average loss: 0.4974, Accuracy: 8492/10000 (85%)

Confusion Matrix

![Screen Shot 2021-08-27 at 1 33 22 pm](https://user-images.githubusercontent.com/43845085/131067769-c7e0d565-aa39-4cde-93e8-3d8a29f8d490.png)

## Part 3 - NetConv 
Final Accuracy
Test set: Average loss: 0.2481, Accuracy: 9387/10000 (94%)

Confusion Matrix

![Screen Shot 2021-08-27 at 1 34 20 pm](https://user-images.githubusercontent.com/43845085/131067858-0550d87b-e811-4571-a595-8fc3aaaafbcd.png)

## Part 4 - Discussion 
It is clear from the results above that the accuracy improves as the complexity of the model increases. We can see that ‘NetLin’ which was the simplest of the three models had the lowest accuracy of around 70% whereas ‘NetFull’, which employs the use of a hidden layer with ‘tanh’ activation performed better with 85% accuracy. Furthermore, NetConv utilised convolutional layers as well as a fully connected layer with ‘relu’ activations which gave us an accuracy of 94% which was the highest out of the three models.

Fig 1.4 – Most frequent misclassifications (red = most frequent, orange = 2nd most frequent, yellow = 3rd most frequent)

![Screen Shot 2021-08-27 at 1 35 20 pm](https://user-images.githubusercontent.com/43845085/131067948-bf606c39-9ba2-4304-b661-cd8e4c909dd9.png)

If we look at ‘NetConv’ and its three most frequent misclassifications in descending order from fig1.4 above, we can see that this model is most likely to mistake:
1. は (ha) for す (su)
2. き (ki) for ま (ma)
3. お (o) for な (na)

All three models seem to misclassify ‘は (ha) for す (su)’ and ‘き (ki) for ま (ma)’. However, ‘NetLin’ and ‘NetFull’ often mistakes ‘お (o)’ for ‘や (ya) and は (ha)’ rather than ‘な (na)’ which is how ‘NetConv’ behaves.

To understand why some characters may be mistaken for others, we look at three comparisons below. We can see that the character ‘は (ha)’ is often mistaken for ‘す (su)’ and ‘き (ki)’ is mistaken for ‘ま (ma)’ in all three models. From looking at the comparisons below we can see two very similar features in both characters circled below for both pairs.

![Screen Shot 2021-08-27 at 1 36 21 pm](https://user-images.githubusercontent.com/43845085/131068011-195765f8-a969-448b-88eb-1dd0697b39fd.png)

For the ‘NetConv’ model, ‘お (o)’ is often mistaken for ‘な (na)’, however the other two models ‘NetLin’ and ‘NetFull’ mistake ‘お (o)’ for either ‘や (ya)’ or ‘は (ha)’. By
inspection we can see that the misclassification from ‘NetConv’ is more convincing compared to the misclassification from ‘NetLin’ and ‘NetFull’ for this example.

![Screen Shot 2021-08-27 at 1 36 52 pm](https://user-images.githubusercontent.com/43845085/131068040-27d41d05-337d-4493-8fb1-9f60b55ec117.png)


