# Cat-in-the-dat

https://www.kaggle.com/competitions/cat-in-the-dat

https://www.kaggle.com/competitions/cat-in-the-dat-ii

In this competition, you will be predicting the probability [0, 1] of a binary target column.

The data contains binary features (`bin_*`), nominal features (`nom_*`), ordinal features (`ord_*`) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features `ord_{3-5}` are lexically ordered according to string.ascii_letters.

Since the purpose of this competition is to explore various encoding strategies, the data has been simplified in that (1) there are no missing values, and (2) the test set does not contain any unseen feature values (See this). (Of course, in real-world settings both of these factors are often important to consider!)