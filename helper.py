import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wandb
from typing import List, Dict, Tuple, Union, Optional

def plot_bar_chart(results: Dict[int, List[float]], x_label: str, y_label: str, plot_title: str,
                   log_wandb: bool = False):
    # Average results
    mean_results = {}
    for split, result in results.items():
        mean_results[split] = sum(result) / len(result)

    # Calculate 1 SEM
    sem_results = {}
    for split, result in results.items():
        sem_results[split] = np.std(result) / math.sqrt(len(result))
    

    # Plot bar chart
    plt.bar(list(results.keys()), list(mean_results.values()), align='center', yerr=list(sem_results.values(), capsize=5),
            capsize=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # TODO: figure out how to set x axis labels to exactly 'splits'
    # plt.xticks()
    plt.ylim((0., 1.))

    if log_wandb:
        wandb.log({plot_title: plt})

    plt.show()