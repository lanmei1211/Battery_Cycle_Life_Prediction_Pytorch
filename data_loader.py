import numpy as np
import pickle
import pprint
import csv
import collections
from torch.utils.data import Dataset

pp = pprint.PrettyPrinter(indent=4)

NUMBAT1 = 41
NUMBAT2 = 43
NUMBAT3 = 40
NUMBAT = 124


def calculate_and_save_scaling_factors(data_dict, train_test_split, csv_dir):
    """Calculates the scaling factors for every feature based on the training set in train_test_split
    and saves the result in a csv file. The factors are used during writing of the tfrecords files."""

    print("Calculate scaling factors...")
    scaling_factors = dict()

    if train_test_split is not None:
        # only take training cells
        data_dict = {k: v for k, v in data_dict.items() if k in train_test_split["train"]}
    else:
        # only take non-secondary-test cells
        data_dict = {k: v for k, v in data_dict.items() if k.startswith('b3')}

    # Calculating max values for summary features
    for k in ['IR',
              'QD',
              'Remaining_cycles',  # The feature "Current_cycles" will be scaled by the same scaling factor
              'Discharge_time']:
        # Two max() calls are needed, one for every cell, one over all cells
        scaling_factors[k] = max([max(cell_v["summary"][k])
                                  for cell_k, cell_v in data_dict.items()
                                  for cycle_v in cell_v["cycles"].values()])

    # Calculating max values for detail features
    for k in ['Qdlin',
              'Tdlin']:
        # Two max() calls are needed, one over every cycle array, one over all cycles (all cells included)
        scaling_factors[k] = max([max(cycle_v[k])
                                  for cell_k, cell_v in data_dict.items()
                                  for cycle_v in cell_v["cycles"].values()])

    with open(csv_dir, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=scaling_factors.keys())
        writer.writeheader()  # Write the field names in the first line of the csv
        writer.writerow(scaling_factors)  # Write values to the corrent fields
    print("Saved scaling factors to {}".format(csv_dir))
    print("Scaling factors: {}".format(scaling_factors))
    return scaling_factors


def load_preprocessed_data():
    """
    Loads a train_test_split dict that divides all cell names into three lists,
    recreating the splits from the original paper.
    This can be passed directly to "write_to_tfrecords()" as an argument.
    """
    # test_ind = np.hstack((np.arange(0, (NUMBAT1 + NUMBAT2), 2), 83))
    # train_ind = np.arange(1, (NUMBAT1 + NUMBAT2 - 1), 2)
    # secondary_test_ind = np.arange(NUMBAT1 - NUMBAT3, NUMBAT)
    # print(test_ind)
    # print(train_ind)
    # print(secondary_test_ind)
    return pickle.load(open('./data/preprocessed_data.pkl', 'rb'))
    # pp.pprint(data)
    # ordered_data = collections.OrderedDict(data)
    # print(list(ordered_data)[test_ind[:]])


# The class inherits the base class Dataset from pytorch
class LoadData(Dataset):  # for training/testing
    def __init__(self, data_path, window):
        super(LoadData, self).__init__()
        self.max_cycles_allowed = 20
        self.nr_stacked_cycles = 4
        self.timeseries_samples = []
        self.scalar_samples = []
        self.targets = []
        self.index = 0
        self.data = pickle.load(open(data_path, 'rb'))
        self.keys = self.data.keys()
        self.window = window
        # self.scaling = calculate_and_save_scaling_factors(self.data, None, 'data/scaling_factors.csv')
        self.scaling = {'IR': 0.021187941, 'QD': 1.0777442, 'Remaining_cycles': 1835.0, 'Discharge_time': 14.4768716666668, 'Qdlin': 1.0614862, 'Tdlin': 43.10046952811214}

        # go through each battery
        for battery in self.data.keys():
            # find the first 16 cycles in the battery by extractng their keys
            cycles = []
            for i, cycle in enumerate(self.data[battery]['cycles']):
                cycles.append(cycle)
                if i == (self.max_cycles_allowed - 1):
                    break

            # loop trough the cycles
            for i, cycle in enumerate(cycles):
                # take next battery when the 16th cycle is reached
                # then we will have 16 stacked data frames of length 4
                if i == (self.max_cycles_allowed - self.nr_stacked_cycles - 1):
                    break

                # extract scalar values
                current_cycle = self.data[battery]['cycle_life'] - \
                    self.data[battery]['summary']['Remaining_cycles'][i + (self.nr_stacked_cycles - 1)]
                self.targets.append([self.data[battery]['summary']['Remaining_cycles'][i + (self.nr_stacked_cycles - 1)], current_cycle])
                self.scalar_samples.append([self.data[battery]['summary']['IR'][i],
                                            self.data[battery]['summary']['QD'][i],
                                            self.data[battery]['summary']['Discharge_time'][i]])

                # extract charge and temperature
                # convert shape from (x,) to (x,1) in order to make it transposable
                # cycle is ALLOWAYS ONE OFF (+1)
                Qdlin_1 = self.data[battery]['cycles'][cycle]['Qdlin']
                Qdlin_1_len = len(self.data[battery]['cycles'][cycle]['Qdlin'])
                Qdlin_1 = Qdlin_1.reshape(Qdlin_1_len, 1)
                Qdlin_2 = self.data[battery]['cycles'][cycles[int(cycle) - 1 + 1]]['Qdlin']
                Qdlin_2 = Qdlin_2.reshape(Qdlin_1_len, 1)
                Qdlin_3 = self.data[battery]['cycles'][cycles[int(cycle) - 1 + 2]]['Qdlin']
                Qdlin_3 = Qdlin_3.reshape(Qdlin_1_len, 1)
                Qdlin_4 = self.data[battery]['cycles'][cycles[int(cycle) - 1 + 3]]['Qdlin']
                Qdlin_4 = Qdlin_4.reshape(Qdlin_1_len, 1)

                Tdlin_1 = self.data[battery]['cycles'][cycle]['Tdlin']
                Tdlin_1_len = len(self.data[battery]['cycles'][cycle]['Tdlin'])
                Tdlin_1 = Tdlin_1.reshape(Tdlin_1_len, 1)
                Tdlin_2 = self.data[battery]['cycles'][cycles[int(cycle) - 1 + 1]]['Tdlin']
                Tdlin_2 = Tdlin_2.reshape(Tdlin_1_len, 1)
                Tdlin_3 = self.data[battery]['cycles'][cycles[int(cycle) - 1 + 2]]['Tdlin']
                Tdlin_3 = Tdlin_3.reshape(Tdlin_1_len, 1)
                Tdlin_4 = self.data[battery]['cycles'][cycles[int(cycle) - 1 + 3]]['Tdlin']
                Tdlin_4 = Tdlin_4.reshape(Tdlin_1_len, 1)

                # stack arrays in sequence horizontally (column wise). This transpose data
                Qdlin_transposed = np.hstack((Qdlin_1, Qdlin_2, Qdlin_3, Qdlin_4))
                Tdlin_transposed = np.hstack((Tdlin_1, Tdlin_2, Tdlin_3, Tdlin_4))

                # stack arrays in sequence depth wise (along third axis).
                stacked_data = np.dstack((Qdlin_transposed, Tdlin_transposed))
                self.timeseries_samples.append(stacked_data)

    def __getitem__(self, idx):
        return self.timeseries_samples[idx], self.targets[idx][0], self.targets[idx][1]

    def __len__(self):
        return len(self.timeseries_samples)
