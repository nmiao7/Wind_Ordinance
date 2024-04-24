import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
import logging
import os
import math
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'



class ODE_MLP(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=512, n_layers=2):
        super(ODE_MLP, self).__init__()
        layers = [nn.Linear(y_dim, n_hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU()])
        layers.append(nn.Linear(n_hidden, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        res = self.net(y)
        return res
    



##############
## Training ##
##############



def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return

def train(dyn_sys_info, m, device, dataset, criterion, epochs, lr, weight_decay):

    # Initialize
    n_store, k  = 100, 0
    ep_num, loss_hist, test_loss_hist = torch.empty(n_store+1,dtype=int), torch.empty(n_store+1), torch.empty(n_store+1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train.to(device), Y_train.to(device), X_val.to(device), Y_val.to(device), X_test.to(device), Y_test.to(device)
    num_train = X_train.shape[0]
    dim, time_step = dyn_sys_info
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    torch.cuda.empty_cache()
    

    # Training Loop
    min_relative_error = 1000000
    for i in range(epochs):
        model.train()
        y_pred = torchdiffeq.odeint(model, X_train, t_eval_point, method="rk4")[-1]
        y_pred = y_pred.to(device)
        optimizer.zero_grad()
        train_loss = criterion(y_pred, Y_train)

        train_loss.backward()
        optimizer.step()
        update_lr(optimizer, i, epochs, args.lr)

        # Save Training and Test History
        if i % (epochs//n_store) == 0 or (i == epochs-1):
            with torch.no_grad():
                model.eval()

                y_pred_val = torchdiffeq.odeint(model, X_val, t_eval_point, method="rk4")[-1]
                val_loss = criterion(y_pred_val, Y_val) * (1 / time_step / time_step)
                y_pred_test = torchdiffeq.odeint(model, X_test, t_eval_point, rtol=1e-9, atol=1e-9, method="rk4")[-1]
                y_pred_test = y_pred_test.to(device)         
                test_loss = criterion(y_pred_test, Y_test)

                if test_loss < min_relative_error:
                    min_relative_error = test_loss
                    # Save the model
                    torch.save(model.state_dict(), f"../test_result/best_model.pth")
                    logger.info(f"Epoch {i}: New minimal relative error: {min_relative_error:.2f}%, model saved.")

                logger.info("Epoch: %d Train: %.5f Test: %.5f", i, train_loss.item(), test_loss.item())
                print("Epoch: ", i, " Train: {:.5f}".format(train_loss.item()), " Test: {:.5f}".format(test_loss.item()))
                ep_num[k], loss_hist[k], test_loss_hist[k] = i, train_loss.item(), test_loss.item()

                k = k + 1

    return ep_num, loss_hist, test_loss_hist, jac_diff_train, jac_diff_test, Y_test



##############
#### Plot ####
##############

def plot_loss(epochs, train, test, path):
    fig, ax = subplots()
    ax.plot(epochs[30:].numpy(), train[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Train")
    ax.plot(epochs[30:].numpy(), test[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Test")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig(path, bbox_inches ='tight', pad_inches = 0.1)


def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    

if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE"])
    parser.add_argument("--dyn_sys", default="lorenz", choices=["lorenz", "rossler"])
    parser.add_argument("--model_type", default="MLP_skip", choices=["MLP","MLP_skip", "CNN", "HigherDimCNN", "GRU"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--train_dir", default="../plot/Vector_field/train_MLPskip_Jac/")

    # Initialize Settings
    args = parser.parse_args()
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    dim = 3
    dyn_sys_info = [dim, args.time_step]
    criterion = torch.nn.MSELoss()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Create Dataset
    df = pd.read_csv('../data/merged_data.csv')
    data_array = df.to_numpy()
    tensor_data = torch.tensor(data_array, dtype=torch.float32) 

    print(tensor_data)

    # Train: X,y

    # Test: X,y

    # Create model
    if args.model_type == "MLP":
        m = ODE_MLP(y_dim=dim, n_hidden=args.n_hidden, n_layers=args.n_layers).to(device)
    elif args.model_type == "MLP_skip":
        m = ODE_MLP_skip(y_dim=dim, n_hidden=args.n_hidden).to(device)

    print("Training...") # Train the model, return node
    epochs, loss_hist, test_loss_hist, jac_train_hist, jac_test_hist, Y_test = train(dyn_sys_info, m, device, dataset, criterion, args.num_epoch, args.lr, args.weight_decay)

    # Plot Loss
    loss_path = f"../plot/Loss/{args.dyn_sys}/{args.model_type}_{args.loss_type}_Total_{start_time}.png"

    plot_loss(epochs, loss_hist, test_loss_hist, loss_path) 
    logger.info("%s: %s", "Training Loss", str(loss_hist[-1]))
    logger.info("%s: %s", "Test Loss", str(test_loss_hist[-1]))
