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

        self.data = pickle.load(open(data_path, 'rb'))
        self.keys = self.data.keys()
        self.window = window
        self.scaling = calculate_and_save_scaling_factors(self.data, None, 'data/scaling_factors.csv')
        # print(self.data.keys())
        # for cell_name, cell_data in self.data.items():
        #    write_single_cell(cell_name, cell_data, data_dir, scaling_factors)

    def __getkeys__(self):
        return self.data.keys(), self.data['b1c2'].keys(), self.data['b1c2']['summary'].keys(), self.data['b1c2']['cycles']['10'].keys()  

    def __getitem__(self, bat_key):
        print(bat_key)
        discharge_timeseries = self.data['b1c2']['cycles']['10']['Qdlin']
        temperature_timeseries = self.data['b1c2']['cycles']['10']['Tdlin']
        return discharge_timeseries

    def __len__(self):
        #Qdlen = len(self.data[bat_key]['cycles'][cycle_nr]['Qdlin'])
        #Vdlen = len(self.data[bat_key]['cycles'][cycle_nr]['Vdlin'])
        #Tdlen = len(self.data[bat_key]['cycles'][cycle_nr]['Tdlin'])
        #return Qdlen, Vdlen, Tdlen
        return 10000
