# cs-457-project
We have included three code files in addition to this read me

function.py holds a bunch of useful function we used to access our data. 

data_creation.ipynb has different code blocks and text showing how we collected and organized our data into various structures so it was easier to access and use

final_cody.py is broken down into two sections. Main analysis and Google analysis which we describe generally below.

MAIN ANALYSIS
In this section our Model is trained on the transcript calls of 50 companies. In order to insert a transcript into BERT we must first preprocess 
them into multiple "text-chunks" and assign each chuck the label of its parent transcript. In order to decide the labels we looked at the price 
movement over 20 days and 60 days. If the price went up by more than 5% over 20 days and more than 10% over 60 days it was label 1. If it went 
down by 5% and 10% then it got the label -1. And if it did neither it got a 0.

After preprocessing we have went from approximately 3,400 transcripts and labels to 187,313 text-chunks and labels.

After trial and error to find what batch size and epoch size would run on the ada cluster we ended up using a mini-batch size of 25 and epoch size 
of 10. From here we tested multiple learning rates but none of them changed the ending accuracy so for the submitted code we have a learning rate or .001.

For our testing data set we used the transcripts of 5 of the 250 companies we didn't use for training and the same preprocessing as described above.

get_tickers() returns the list of 503 stock tickers for all of the companies we collected price data on

get_stock_dict() returns to us a dictionary of dataframes, where every key is a companies ticker and the value is a dataframe holding all of their stock data

get_companies_by_50 returns to us 5 different embedded dictionaries. Each dictionary holds 50 companies where the key is the ticker name for each company,
then the value is another dictionary for each of the years the selected company has released a earnings call. The next dictionary is the quarter and then 
the next one holds the exact date/time the transcript was released and the text file of the transcript.


GOOGLE ANALYSIS
In this section, we train BERT only on Google earnings call transcripts to determine whether BERT is better suited to company-specific predictions.

The key difference in the data are the criteria for labeling. Unlike the 5% (1-month) and 10% (3-month) thresholds in the main analysis, here we use
1% and 3% thresholds.

We use the same data processing strategy as the main analysis. Google's 69 available transcripts are split into text chunks. Those text chunks are individual
observations. Those observations are split 80-10-10 into train, development, and test sets.

Then we look for the hyperparameters to use. We repeatedly train BERT on the dev set with different learning rates and batch sizes, with 3 epochs per pairing. 
We save and use the learning rate and batch size that correspond to the lowest 3rd epoch loss.

For our analysis, we found a learning rate of 2e-05 and a batch size of 16 corresponded to the lowest loss.

We train BERT with those hyperparameters and 10 epochs, a value set more by time constraints than careful thought. It may have benefited us, however, since 
in previous experiments with higher epochs, we had much worse overfitting.

GOOG_transcripts.csv, AAPL_transcripts.csv, and MSFT_transcripts.csv are just from a slightly modified version of our previously defined data processing code 
and turned into .csv files with the pandas to_csv function. Specific modifications to the data processing code are to remove the for loop and define 
[company = t_50["GOOG"]]. This also works for AAPL (Apple) and MSFT (Microsoft) because they are in the list of the first 50 companies.

data_preprocess() takes a dataframe and outputs data ready to be loaded into a dataloader.

train_model() takes a learning rate, a batch size, a number of epochs, and data (output of data_preprocess()) and trains BERT. In our code, BERT was already initailized outside the function.
predict() takes text and uses BERT to output probabilities and a predicted label.

getBERTScores() takes a dataframe and adds adds "neutral," "bad," "good," and "predict" features to it. The first three new features correspond to probabilities for each label, while the "predict" feature is the predicted label.