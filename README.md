# textData
This repository is created with the purpose of showcasing my ability to manipulate, organize, and simplify data using Python's pandas and scikit-learn packages.

In this project I clean and reduce text data from a recent Kaggle competition:
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

The given data is (mostly) raw text from internet comments, which was collected for the purpose of measuring "toxicity".  Each comment is labeled with a measure (float point number between 0 and 1) of its toxicity, based on the results of a survey of human raters.

My objective here was to convert the data to a bag-of-words structure for use in a regression model.  I also looked to improve the structure's vocabulary using feature reduction techniques.
