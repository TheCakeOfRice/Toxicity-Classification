import numpy as np
import pickle
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from query_filler import QueryFiller, targets_to_fill_labels
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from time import time


class ToxicityMeasurer:
	def __init__(
		self,
		C=0.5,
		BOW_vect=None,
		query_filler=None
	):
		self.BOW_vect = BOW_vect
		self.query_filler = query_filler
		self.logit = LogisticRegression(
			C=C,
			solver="liblinear",
			max_iter=1000
		)

	def fit(self, X, y):
		"""
		Fits the model to the data X, given targets y.
		"""
		print("Fitting logistic regression...")
		start = time()
		self.logit.fit(X, y)
		end = time()
		minutes = (end - start) // 60
		seconds = (end - start) % 60
		print(
			"Fitting finished in " +
			str(minutes) +
			" minutes and " +
			str(seconds) +
			" seconds."
		)

	def predict_target(
		self,
		X,
		process_from_raw_text=False,
		fill_from_BOW=False
	):
		"""
		Predicts the target of X.  Options include:
			- process_from_raw_text: Set True if X is an iterable of strings
			- fill_from_BOW: Set True if X is a bag-of-words representation
				of your test data
		"""
		if process_from_raw_text:
			X = self.BOW_vect.transform(X)
			fill_from_BOW = True
		if fill_from_BOW and self.query_filler is not None:
			X = sparse.hstack([
				X,
				sparse.csr_matrix(self.query_filler.fill(X))
			]).tocsr()

		return self.logit.predict_proba(X)

	def predict_label(
		self,
		X,
		process_from_raw_text=False,
		fill_from_BOW=False
	):
		"""
		Predicts the label of X.  Options include:
			- process_from_raw_text: Set True if X is an iterable of strings
			- fill_from_BOW: Set True if X is a bag-of-words representation
				of your test data
		"""
		if process_from_raw_text:
			X = self.BOW_vect.transform(X)
			fill_from_BOW = True
		if fill_from_BOW and self.query_filler is not None:
			X = sparse.hstack([
				X,
				sparse.csr_matrix(self.query_filler.fill(X))
			]).tocsr()

		return self.logit.predict(X)

	def cv_score(self, X, y, cv=1, average=True):
		"""
		Cross validates the model using data X and targets y.  Calculates accuracy
		on each fold.  Options include:
			- cv: The number of folds for cross-validation.  If cv=1, returns the
				training accuracy.
			- average: Boolean variable which dictates if results should be averaged.
				If True, returns a float instead of an array of floats.
		"""
		scores = cross_val_score(
			self.logit,
			X,
			y,
			cv=cv
		)
		if average or cv == 1:
			return np.mean(scores)
		else:
			return scores


def target_to_label(target):
	"""
	Converts an array of targets to their corresponding labels:
		(target >= 0.5) --> 1, "Toxic"
		(target < 0.5) --> -1, "Not Toxic"
	"""
	label = np.ones(len(target), dtype=np.int32)
	for i in range(len(target)):
		if target[i] < 0.5:
			label[i] = -1

	return label


if __name__ == "__main__":
	start = time()
	print("== Running model.py ==")

	# Importing saved data
	with open("variables/BOW_vect.pickle", "rb") as file:
		BOW_vect = pickle.load(file)
	BOW_train = sparse.load_npz("variables/training/BOW_train.npz")
	target_train = np.load("variables/training/target_train.npy")

	# Labeling samples based on target value
	label_train = target_to_label(target_train)

	# Declaring and fitting our model
	model = ToxicityMeasurer(
		C=0.25,
		BOW_vect=BOW_vect
	)
	model.fit(BOW_train, label_train)
	with open("variables/model.pickle", "wb") as file:
		pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

	# # Tuning the parameter C to give the best generalization possible
	# # using 5-fold cross-validation
	# # -- Initial calculations
	# Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	# CV_acc = []
	# for C in Cs:
	# 	print("Computing CV accuracy for C = " + str(C) + "...")
	# 	model.logit.C = C
	# 	CV_acc += [model.cv_score(
	# 		BOW_train,
	# 		label_train,
	# 		cv=5,
	# 		average=True
	# 	)]

	# # -- Plotting results
	# ticks = list(range(len(Cs)))
	# plt.title(r"5-fold CV accuracy versus regularization amount $C$")
	# plt.xlabel(r"$C$")
	# plt.ylabel("Average classification accuracy")
	# plt.xticks(ticks, labels=Cs)
	# plt.ylim(0.5, 1)
	# plt.bar(ticks, CV_acc)
	# plt.savefig("graphs/C_tuning.png", dpi=500)

	# # -- Zoomed-in calculations
	# Cs = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75]
	# CV_acc = []
	# for C in Cs:
	# 	print("Computing CV accuracy for C = " + str(C) + "...")
	# 	model.logit.C = C
	# 	CV_acc += [model.cv_score(
	# 		BOW_train,
	# 		label_train,
	# 		cv=5,
	# 		average=True
	# 	)]

	# # -- Plotting zoomed-in results
	# ticks = list(range(len(Cs)))
	# plt.title(r"5-fold CV accuracy versus regularization amount $C$")
	# plt.xlabel(r"$C$")
	# plt.ylabel("Average classification accuracy")
	# plt.xticks(ticks, labels=Cs)
	# plt.ylim(0.795, 0.810)
	# plt.bar(ticks, CV_acc)
	# plt.savefig("graphs/C_tuning_zoomed.png", dpi=500)

	# # Testing for overfitting using 5-fold cross-validation
	# # -- Calculating errors
	# acc = model.logit.score(BOW_train, label_train)
	# CV_acc = model.cv_score(BOW_train, label_train, cv=5, average=False)

	# # -- Plotting results
	# ticks = [1, 2, 3, 4, 5]
	# plt.title("Accuracy after 5-fold cross-validation")
	# plt.xlabel("Fold")
	# plt.ylabel("Accuracy")
	# plt.xticks(ticks)
	# plt.ylim(0.9, 1)
	# plt.plot(ticks, acc * np.ones(5), label="Training accuracy", color="red")
	# plt.bar(ticks, CV_acc, label="CV accuracy")
	# plt.legend()
	# plt.savefig("graphs/CV_acc.png", dpi=500)

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
	# plt.show()
