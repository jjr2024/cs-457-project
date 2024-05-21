# cs-457-project
MAIN ANALYSIS





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