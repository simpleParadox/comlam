import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
#Change path here
participant = 1004
path = f'G:\comlam\\results\permuted\\P{participant}\\'
results = []
for f in glob.glob(path + "*.npz"):
    temp = np.load(f, allow_pickle=True)['arr_0'].tolist()
    results.append(temp[1004])
print(np.mean(results))

sns.histplot(results)
plt.title(f"{participant} permuted accs")
plt.xlabel("Accuracies")
plt.show()