from kmeans import KMeansMachine, KMeansTrainer
import timeit
import bob.learn.em
import numpy

def old_bob(data):
	kmeans_machine = bob.learn.em.KMeansMachine(3, 2	)
	kmeans_trainer = bob.learn.em.KMeansTrainer()
	max_iterations = 200
	convergence_threshold = 1e-5
	# Train the KMeansMachine
	bob.learn.em.train(kmeans_trainer, kmeans_machine, data,
	    max_iterations=max_iterations,
	    convergence_threshold=convergence_threshold)
	print(kmeans_machine.means)

def new_bob(data):
	kmeans_machine = KMeansMachine(3, 2)
	kmeans_trainer = KMeansTrainer()
	max_iterations = 200
	convergence_threshold = 1e-5
	# Train the KMeansMachine
	bob.learn.em.train(kmeans_trainer, kmeans_machine, data,
	    max_iterations=max_iterations,
	    convergence_threshold=convergence_threshold)
	print(kmeans_machine.means)

l = [[1, 1, 1], [0, 5, 1], [-2, -2, 1]]

data = []

for el in l:
	arr = numpy.array([numpy.random.normal(el[0], el[2], (40)), numpy.random.normal(el[1], el[2], (40))])
	data.append(arr)

data = numpy.concatenate(data, axis=1).T
# Create a kmeans m with k=2 clusters with a dimensionality equal to 3
old_bob(data)
new_bob(data)

common_setup = """
import numpy
import bob.learn.em
import kmeans

max_iterations = 200
convergence_threshold = 1e-5
l = [[1, 1, 1], [0, 5, 1], [-2, -2, 1]]

data = []

for el in l:
	arr = numpy.array([numpy.random.normal(el[0], el[2], (40)), numpy.random.normal(el[1], el[2], (40))])
	data.append(arr)

data = numpy.concatenate(data, axis=1).T
"""

fun = """
bob.learn.em.train(kmeans_trainer, kmeans_machine, data,
    max_iterations=max_iterations,
	convergence_threshold=convergence_threshold)
"""

setup_old = common_setup + """
kmeans_machine = bob.learn.em.KMeansMachine(3, 2)
kmeans_trainer = bob.learn.em.KMeansTrainer()
"""

setup_new = common_setup + """
kmeans_machine = kmeans.KMeansMachine(3, 2)
kmeans_trainer = kmeans.KMeansTrainer()
"""


print("Old: ", timeit.timeit(fun, setup=setup_old, number=1000))
print("New: ", timeit.timeit(fun, setup=setup_old, number=1000))