import numpy as np
import pandas as pd
import pickle
from time import time
from scipy import sparse
from sklearn.metrics import accuracy_score
from model import ToxicityMeasurer, target_to_label


start = time()
print("== Running test_model.py ==")


with open("variables/model.pickle", "rb") as file:
	model = pickle.load(file)

# # Testing on full dataset/test dataset
# # -- Loading data
# BOW = sparse.load_npz("variables/BOW.npz")
# target = pd.read_csv("variables/target.csv")["target"]\
# 	.values.astype(np.float64)
# # BOW_test = sparse.load_npz("variables/test/BOW_test.npz")
# # target_test = np.load("variables/test/target_test.npy")

# # -- Converting targets to labels
# label = target_to_label(target)

# # -- Predicting labels and targets using the model
# label_pred = model.predict_label(BOW, fill_from_BOW=False)
# target_pred = model.predict_target(BOW, fill_from_BOW=False)[:, 1]

# # -- Saving scores
# model_acc = accuracy_score(label, label_pred)

# # -- Predicting randomly, multiple times for consistency
# random_acc = []
# for i in range(10):
# 	random = np.random.choice([-1, 1], size=BOW.shape[0])
# 	random_acc += [accuracy_score(label, random)]

# # -- Saving random classifier scores
# random_acc = np.mean(random_acc)

# # -- Printing results
# print(
# 	"The model had a mean accuracy of " +
# 	str(model_acc) +
# 	"."
# )
# print(
# 	"A random classifier had a mean accuracy of " +
# 	str(random_acc) +
# 	"."
# )


# Testing on intuitive examples
# -- Declaring examples
X = [
	"Have a great day!",
	"I don't understand how you're so stupid.",
	"Stop feeding you fucking pleb.",
	"You're the worst fucking bot I've ever seen.",
	"Obama was the first black president."
]
y = [
	-1,
	1,
	1,
	1,
	-1
]
console_text = "Processing the comment(s):\n"
for i in range(len(X)):
	console_text += str(i) + ' :    "' + X[i] + '"\n'
print(console_text)

# -- Predicting labels and targets using the model
label_pred = model.predict_label(X, process_from_raw_text=True)
target_pred = model.predict_target(X, process_from_raw_text=True)[:, 1]

# -- Printing results
console_text = "Result(s):\n         label    |    correct\n" +\
	"---------------------------------------\n"
for i in range(len(X)):
	if label_pred[i] == 1:
		is_toxic = "  toxic  "
	elif label_pred[i] == -1:
		is_toxic = "not toxic"
	if label_pred[i] == y[i]:
		correct = "    y    "
	else:
		correct = "    n    "
	console_text += str(i) + " :    " + is_toxic + "  |   " + correct + "\n"
print(console_text)


# Data visualization
# to do


end = time()
minutes = (end - start) // 60
seconds = (end - start) % 60
print(
	"Finished in " +
	str(minutes) +
	" minutes and " +
	str(seconds) +
	" seconds."
)
