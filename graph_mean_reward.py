import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filename = "progress-adamepsbigfish.csv"

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    key_index = -1

    mean_reward = []
    timesteps_end = []
    for line_num, row in enumerate(csv_reader):
        if line_num == 0:
            continue
        row = [x.lower() for x in row]
        mean_reward.append(float(row[2]))
        timesteps_end.append(float(row[6]))

    plt.title('PPO BigFish')
    plt.xlabel('Training Timestep')
    plt.ylabel('Mean Reward')
    plt.plot(timesteps_end, mean_reward)
    plt.show()
