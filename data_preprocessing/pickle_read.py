import sys
import pickle as pkl
import matplotlib.pyplot as plt

# In order to find the folder where the pkl files are
sys.path.append('./..')


def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass


def main():
    with open("data/batch0.pkl", "rb") as f:
        for event in pickleLoader(f):
            break
    # Plot the discharge capacity of the 10th cycle of the first battery in the
    # first batch againt the voltage of the same battery
    plt.plot(event['b0c2']['cycles']['10']['Qd'], event['b0c2']['cycles']['10']['V'])
    plt.xlabel('Cycle Number')
    plt.ylabel('Discharge Capacity (Ah)')
    # Plot the discharge capacity of the 10th cycle of the first battery in the
    # first batch againt the voltage of the same battery
    plt.plot(event['b0c2']['summary']['cycle'], event['b0c2']['summary']['QD'])
    plt.show()


if __name__ == '__main__':
    main()
