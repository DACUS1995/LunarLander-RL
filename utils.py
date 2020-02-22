from matplotlib import pyplot as plt
from typing import List, Tuple, Dict

def plot_final_results(results: Dict):
	num_subplots = len(results)
	order = 1

	for key in results:
		plt.subplot(num_subplots, 1, order)
		plt.plot(results[key])
		plt.ylabel(key)
		order += 1

	plt.xlabel("Episodes")
	plt.show()