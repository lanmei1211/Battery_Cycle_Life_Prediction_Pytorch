import sys
import numpy as np
import pickle

# In order to find the folder where the pkl files are
sys.path.append('./..')


def pkl_to_dict():

    print('Converting the pkl files to a dictionary...')

    batches_dict = {}  # Initializing

    # ------------------------------- #
    # ----------- BATCH 1 ----------- #
    # ------------------------------- #
    batch1 = pickle.load(open(r'data/batch1.pkl', 'rb'))

    # remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    numBat1 = len(batch1.keys())
    assert numBat1 == 41

    # ------------------------------- #
    # ----------- BATCH 2 ----------- #
    # ------------------------------- #

    batch2 = pickle.load(open(r'data/batch2.pkl', 'rb'))

    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]

    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

    del batch2['b2c7']
    del batch2['b2c8']
    del batch2['b2c9']
    del batch2['b2c15']
    del batch2['b2c16']

    numBat2 = len(batch2.keys())
    assert numBat2 == 43

    # Add batch1 dictionary to dictionary
    batches_dict.update(batch1)
    # Add batch2 dictionary to dictionary
    batches_dict.update(batch2)

    # ------------------------------- #
    # ----------- BATCH 3 ----------- #
    # ------------------------------- #

    batch3 = pickle.load(open(r'data/batch3.pkl', 'rb'))

    # remove noisy channels from batch3
    del batch3['b3c37']
    del batch3['b3c2']
    del batch3['b3c23']
    del batch3['b3c32']
    del batch3['b3c38']
    del batch3['b3c39']

    numBat3 = len(batch3.keys())
    assert numBat3 == 40

    # Add batch1 dictionary to dictionary
    batches_dict.update(batch3)

    # ------------------------------- #
    # ------------  PLOT  ----------- #
    # ------------------------------- #

    numBat = numBat1 + numBat2 + numBat3
    assert numBat == 124

    bat_dict = {**batch1, **batch2, **batch3}
    assert len(bat_dict.keys()) == 124

    #test_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
    #train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
    #secondary_test_ind = np.arange(numBat - numBat3, numBat)
    print('DONE')

    return bat_dict


def main():
    pkl_to_dict()


if __name__ == '__main__':
    main()
