from data_loader import LoadData, load_preprocessed_data
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    print('\n\n\tTrain...\n')
    data = load_preprocessed_data()
    dataset = LoadData('./data/preprocessed_data.pkl', window=20)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=True,
                                             pin_memory=True)


    print(dataset.__getkeys__())
    #print(dataset.__len__('b1c2', '10'))
    #print(type(dataset.__getitem__('b1c2', '10')))

    #torch.from_numpy(dataset.__getitem__('b1c2', '10'))
    #torch.from_numpy(dataset.__getitem__('b1c2', '10'))
    #torch.from_numpy(dataset.__getitem__('b1c2', '10'))

    pbar = tqdm(enumerate(dataloader), total=10000)
    for i, (ttt) in pbar:  # batch --------
        #print(ttt)
        # Run model
        # pred = model(imgs)

        # Compute loss
        #loss, loss_items = compute_loss(pred, targets, model, not prebias)
        #if not torch.isfinite(loss):
        #    print('WARNING: non-finite loss, ending training ', loss_items)
        #    return results

        #else:
        #    loss.backward()
        pass

    #lt.show()
    #print(dataset.__getitem__('b1c2'))
    #print(dataset['b1c2']['cycle_life'])
    #print(dataset['b1c2']['summary'].keys())
    #print(dataset['b1c2']['cycles']['1'].keys())
    #print(dataset['b1c2']['cycles']['1']['Qdlin'])
    #print(dataset['b1c2']['cycles']['1']['Vdlin'])
    #for battery in dataset:
    #   print(battery)
    #print(dataset.keys)


if __name__ == "__main__":
    main()
