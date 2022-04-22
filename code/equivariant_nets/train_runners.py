import numpy as np
import torch
from torch import nn
from contextlib import nullcontext
from tqdm import tqdm
import traceback
from .pytorch_helpers import reset_running_stats


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_epoch(loader, net, pp, device, epoch_ctr, opt=None, loss_func=None, lr=0.0):
    net.train() if opt else net.eval()
    mode_str = opt if isinstance(opt, str) else 'train' if opt else 'test'
    if opt == 'bnorm':
        opt = None
    elif opt:
        for pg in opt.param_groups: pg['lr'] = lr
        epoch_ctr[0] += 1
    desc_str = f' {mode_str:5} ep{epoch_ctr[0]:3}, lr={lr:8.6f}, err='

    corr = elem = 0
    with nullcontext() if opt else torch.no_grad():
        with tqdm(total=len(loader), desc=f"{desc_str} -.---%", leave=True) as pbar:
            for x, t in loader:
                x = x.to(device) if not isinstance(x, list) else [t.to(device) for t in x]
                t = t.to(device)
                if opt:
                    opt.zero_grad()
                y = net(x if pp is None else pp(x))
                loss = loss_func(y, t)
                if opt:
                    loss.backward()
                    opt.step()
                corr += (y.detach().argmax(dim=1) == t).sum().item()
                elem += len(t)
                pbar.update(1)
                pbar.set_description(f'{desc_str}{1 - (corr / elem):7.3%}, curloss={loss.item():6.4f}')
            return corr / elem


def train_and_test(model, epochs, loss, opt, lr, train, test, test_pps, train_pp=nn.Identity, device=default_device):
    net = model().to(device)
    opt = opt(net.parameters())
    net_params = dict(net=net, device=device, epoch_ctr=[0], loss_func=loss().to(device))
    for i in range(epochs):
        run_epoch(train, pp=train_pp(), opt=opt, lr=lr(i) if callable(lr) else lr, **net_params)
        getattr(train.dataset, 'next_epoch', lambda: None)()
    n_reset = len(reset_running_stats(net))
    if n_reset:
        print(f' Reset running stats of {n_reset} layers, calculate batch stats...')
        run_epoch(train, opt='bnorm', pp=train_pp(), **net_params)
    print(' Training finished. Running tests...')
    test_acc = [run_epoch(test, pp=pp(), **net_params) for _, pp in test_pps]
    print(' final test errors:', ', '.join(f'{n}{1 - x:8.3%}' for (n, _), x in zip(test_pps, test_acc)))
    return net, test_acc


def run_multi(n_runs, save_file, acc=[], test_pps=[('', nn.Identity)], **kwargs):
    net = None
    try:
        for i in range(n_runs):
            print(f'\nRUN {i + 1} / {n_runs}:')
            net, run_acc = train_and_test(test_pps=test_pps, **kwargs)
            if i == 0 and save_file: torch.save(net.state_dict(), save_file)
            acc.append(run_acc)
    except: traceback.print_exc()
    try:
        if acc:
            print('test err mean value:', ', '.join(f'{n}{x:8.3%}' for (n,_),x in zip(test_pps,1-np.mean(acc,axis=0))))
            print('test err std ddof=1:', ', '.join(f'{n}{x:8.3%}' for (n,_),x in zip(test_pps,np.std(acc,ddof=1,axis=0))))
    except: traceback.print_exc()
    return net, acc
