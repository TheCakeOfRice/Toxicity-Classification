import numpy as np
import pandas as pd
import sqlite3
from scipy import sparse
from sklearn.utils import resample
from time import time


start = time()
print("== Running resampling.py ==")


BOW = sparse.load_npz("variables/BOW.npz")
target = pd.read_csv("variables/target.csv")["target"]\
	.values.astype(np.float64)
other_cols = pd.read_csv("variables/other_cols.csv").drop("id", axis=1)\
	.values.astype(np.float64)


# Grabbing indicies based on sample label
print("Grabbing indices...")
toxic_ind = np.where(target >= 0.5)[0]
not_toxic_ind = np.where(target < 0.5)[0]


# Resampling to create balanced classes,
# and also for computational considerations
print("Resampling...")
BOW_toxic, target_toxic, other_cols_toxic = resample(
	BOW[toxic_ind],
	target[toxic_ind],
	other_cols[toxic_ind],
	replace=False,
	n_samples=100000
)
BOW_not_toxic, target_not_toxic, other_cols_not_toxic = resample(
	BOW[not_toxic_ind],
	target[not_toxic_ind],
	other_cols[not_toxic_ind],
	replace=False,
	n_samples=100000
)


# Merging results into training, test, and calibration sets
print("Merging results...")
BOW_train = sparse.vstack([
	BOW_toxic[0:50000],
	BOW_not_toxic[0:50000]
])
BOW_test = sparse.vstack([
	BOW_toxic[50000:75000],
	BOW_not_toxic[50000:75000]
])
BOW_calib = sparse.vstack([
	BOW_toxic[75000:100000],
	BOW_not_toxic[75000:100000]
])

target_train = np.append(
	target_toxic[0:50000],
	target_not_toxic[0:50000]
)
target_test = np.append(
	target_toxic[50000:75000],
	target_not_toxic[50000:75000]
)
target_calib = np.append(
	target_toxic[75000:100000],
	target_not_toxic[75000:100000]
)

other_cols_train = np.vstack([
	other_cols_toxic[0:50000],
	other_cols_not_toxic[0:50000]
])
other_cols_test = np.vstack([
	other_cols_toxic[50000:75000],
	other_cols_not_toxic[50000:75000]
])
other_cols_calib = np.vstack([
	other_cols_toxic[75000:100000],
	other_cols_not_toxic[75000:100000]
])


# Saving results
print("Saving results...")
sparse.save_npz("variables/training/BOW_train.npz", BOW_train)
sparse.save_npz("variables/test/BOW_test.npz", BOW_test)
sparse.save_npz("variables/calibration/BOW_calib.npz", BOW_calib)

np.save("variables/training/target_train.npy", target_train)
np.save("variables/test/target_test.npy", target_test)
np.save("variables/calibration/target_calib.npy", target_calib)

np.save("variables/training/other_cols_train.npy", other_cols_train)
np.save("variables/test/other_cols_test.npy", other_cols_test)
np.save("variables/calibration/other_cols_calib.npy", other_cols_calib)


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
