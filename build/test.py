import kmeans
from matplotlib import pyplot as plt
import numpy as np

l = [[1, 1, 1], [0, 5, 1], [-2, -2, 1]]

data = []

for el in l:
	arr = np.array([np.random.normal(el[0], el[2], (40)), np.random.normal(el[1], el[2], (40))])
	data.append(arr)

data = np.concatenate(data, axis=1).T
print(data.shape)

plt.scatter(data[:, 0], data[:, 1])
plt.show()

trainer = kmeans.KMeansTrainer()
machine = kmeans.KMeansMachine(3, 2)
trainer.initialize(machine, data)

for i in range(10):
	means = machine.getMeans()
	plt.scatter(data[:, 0], data[:, 1])
	print(means)
	plt.scatter(means[:, 0], means[:, 1])
	plt.show()

	trainer.eStep(machine, data)
	trainer.mStep(machine)