# Cyberbullying Detection and Classification
In this repository, I've trained and tested some algorithms for my post-graduate program Data and Knowledge Engineering in Hacettepe University.

## Purpose
Cyberbullying is a gigantic problem in these years. **In 2019, 70% of the people 15 to 24 years old** who are online globally
being attacked by cyberbullies. To stop these attacks, we need to take some precautions.

In my project, I tried to use artificial intelligence and neural networks to detect cyberbullies from their actions.
I focused on the Twitter because there is amazingly huge data to be processed (**350,000 Tweets per minute**).

A social media user, Twitter in this case, can be classified as cyberbully or not from his tweet history.

In my training data, there are 6 categories to classify the user: **Age, Ethnicity, Religion, Gender, Other Cyberbullying and Not**

For the sake of calculation, I didn't use the *Other Cyberbullying* because this category cannot be put either one side or the other.


## Installation & Running
There are only few steps to install necessary libraries and run the algorithms. After downloading the code, you just need to run following commands.

```
$> python3 -m venv venv
$> . venv/bin/activate
$> pip install -r requirements.txt
```

If you are going to use Jupyter Notebooks, then you need to run these codes as well:
```
$> ipython kernel install --user --name=new_venv
$> pip install -r requirements.txt
```
After running these commands, you can change the *kernel* from notebook screen.

## Results
There are 3 main python scripts. Naive Bayes and LSTM algorithms are different scripts.
The other algorithms are in a different file. You can run all the algorithms as you wish.


You can see the results from the comparison table:

![image](https://user-images.githubusercontent.com/18647074/168083448-6c7eba1b-e770-482f-bc25-d51062bbceb9.png)


The LSTM algorithm gave the best results in these tests. I'd prefer to choose Naive Bayes and Perceptron algorithm as the second best due to their accuracy. 

Version 1.0.1

