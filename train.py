from data_loader import LoadData, load_preprocessed_data
from models import Net
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse


def train(args, model, device, train_loader, optimizer, epoch):
    tb_writer = SummaryWriter()
    model.train()
    nb = train_loader.__len__()
    pbar = tqdm(enumerate(train_loader), total=nb)
    for batch_idx, (timeseries_data, remaining, current) in pbar:  # batch --------
        #print(timeseries_data.shape)
        #print(timeseries_data)
        #print(remaining.shape)
        data, remaining_t, _ = timeseries_data.to(device, dtype=torch.float), remaining.to(device, dtype=torch.float), current.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, remaining_t)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        #print('cols', len(timeseries_data[0]))
        #print(timeseries_data.shape)
        #rint(remaining.shape)
        #rint(current.shape)
        #rint(remaining)
        #rint(current)
        #print()
    # Write Tensorboard results
    if tb_writer:
        x = [loss]
        titles = ['MSE']
        for xi, title in zip(x, titles):
            tb_writer.add_scalar(title, xi, epoch)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Battery')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    print('\n\n\tTrain...\n')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # data = load_preprocessed_data()
    dataset = LoadData('./data/preprocessed_data.pkl', window=20)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               num_workers=1,
                                               shuffle=True,
                                               pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        print('\tEPOCH', epoch, '/', args.epochs)
        train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()
