import pickle as pkl
import matplotlib.pyplot as plt


def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass


def main():
    with open("./batch0.pkl", "rb") as f:
        for event in pickleLoader(f):
            print(event)
            break
    plt.plot(event['b0c1']['cycles']['10']['Qd'], event['b0c1']['cycles']['10']['V'])
    plt.plot(event['b0c1']['summary']['cycle'], event['b0c1']['summary']['QD'])
    plt.show()


if __name__ == '__main__':
    main()
