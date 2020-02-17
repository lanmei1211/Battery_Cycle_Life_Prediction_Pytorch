from data_loader import LoadData, load_preprocessed_data
import torch
import matplotlib.pyplot as plt


def main():
    print('\n\n\tTrain...\n')
    data = load_preprocessed_data()
    dataset = LoadData('./data/preprocessed_data.pkl')
    # pp.pprint(dataset)
    #print(dataset.__getitem__('b1c2').key())
    #print(dataset.__getitem__.key())
    print(dataset.__getkeys__)
    print(type(dataset.__getitem__('b1c2')['cycles']['10']['Qdlin']))
    #plt.plot(dataset.__getitem__('b1c2')['cycles']['10']['Qdlin'],
    #         dataset.__getitem__('b1c2')['cycles']['10']['Vdlin'])
    torch.from_numpy(dataset.__getitem__('b1c2')['cycles']['10']['Qdlin'])
    torch.from_numpy(dataset.__getitem__('b1c2')['cycles']['10']['Vdlin'])
    torch.from_numpy(dataset.__getitem__('b1c2')['cycles']['10']['Tdlin'])

    

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
