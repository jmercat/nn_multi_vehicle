import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from utils.losses import maskedMSE
from utils.utils import Settings, make_dir
from utils.utils import get_multi_object_dataset, get_multi_object_net

args = Settings()

make_dir(args.log_path + args.model_type + '/')
make_dir(args.models_path + args.model_type + '/')
logger = SummaryWriter(args.log_path + args.name)

# logger.add_hparams(args.get_dict(), {})

trSet, valSet = get_multi_object_dataset()

net = get_multi_object_net()

if args.optimizer == 'Ranger':
    optimizer = Ranger(net.parameters(), lr=args.lr, alpha=0.5, k=5, weight_decay=1e-3)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-3)

trDataloader = DataLoader(trSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=valSet.collate_fn)

# torch.autograd.set_detect_anomaly(True)
iter_num = 0
for epoch_num in range(args.n_epochs):

    it_trDataloader = iter(trDataloader)
    it_valDataloader = iter(valDataloader)

    len_tr = len(it_trDataloader)
    len_val = len(it_valDataloader)

    net.train_flag = True

    avg_mse_loss = 0
    avg_loss = 0

    for i in range(len_tr):
        # start_time = timer()
        iter_num += 1
        data = next(it_trDataloader)
        hist = data[0].to(args.device)
        fut = data[1].to(args.device)
        mask_hist = data[2].to(args.device)
        mask_fut = data[3].to(args.device)

        optimizer.zero_grad()
        # data_time = timer()
        # print('Time getting data: ', data_time - start_time)
        fut_pred = net(hist)[-fut.shape[0]:]
        # pred_time = timer()
        # print('Time prediction: ', pred_time - data_time)
        mask = torch.cumprod(1 - (fut[:, :, 0, 0] == 0).float() * (fut[:, :, 0, 1] == 0).float(), dim=0)
        #ego mse only
        mse_loss = maskedMSE(fut_pred[:, :, 0, :], fut[:, :, 0, :], mask, 2)

        loss = mse_loss
        if loss != loss:
            print('Nan')
            continue
            # raise RuntimeError("The loss value is Nan.")
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        # step_time = timer()
        # print('Time backward: ', step_time - pred_time)

        avg_mse_loss += mse_loss.detach()
        avg_loss += loss.detach()
        # print('Overall step time: ', step_time - start_time)

        if i%args.print_every_n == args.print_every_n-1:
            torch.save(net.state_dict(), args.models_path + args.name + '.tar')
            avg_loss = avg_loss.item()

            print("Epoch no:", epoch_num + 1, "| Epoch progress(%):",
                  format(i / (len(trSet) / args.batch_size) * 100, '0.2f'),
                  "| loss:", format(avg_loss / args.print_every_n, '0.4f'),
                  "| MSE:", format(avg_mse_loss / args.print_every_n, '0.4f'))
            info = {'loss': avg_loss/args.print_every_n, 'mse': avg_mse_loss / args.print_every_n}

            for tag, value in info.items():
                logger.add_scalar(tag, value, int((epoch_num*len_tr + i)/args.print_every_n))
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if len(param.data) > 1:
                        pass
                        # logger.add_histogram(name[1:], param.data, int((epoch_num*len_tr + i)/args.print_every_n))
                        # logger.add_histogram(name[1:] + '_grad', param.grad.data, int((epoch_num*len_tr + i)/args.print_every_n))
                    else:
                        try:
                            logger.add_scalar(name[1:], param.data.squeeze()[0], int((epoch_num * len_tr + i) / args.print_every_n))
                            # logger.add_scalar(name[1:] + '_grad', param.grad.data.squeeze()[0],
                            #                  int((epoch_num * len_tr + i) / args.print_every_n))
                        except:
                            logger.add_scalar(name[1:], param.data,
                                              int((epoch_num * len_tr + i) / args.print_every_n))
                            # logger.add_scalar(name[1:] + '_grad', param.grad.data,
                            #                   int((epoch_num * len_tr + i) / args.print_every_n))
            avg_mse_loss = 0
            avg_loss = 0

    torch.save(net.state_dict(), args.models_path + args.model_type + '/' + args.name + '.tar')
    avg_loss = 0
    net.train_flag = False
    for j in range(len_val):
        data = next(it_valDataloader)
        hist = data[0].to(args.device)
        fut = data[1].to(args.device)
        mask_hist = data[2].to(args.device)
        mask_fut = data[3].to(args.device)

        fut_pred = net(hist)[-fut.shape[0]:]
        mask = torch.cumprod(1 - (fut[:, :, 0, 0] == 0).float() * (fut[:, :, 0, 1] == 0).float(), dim=0)
        mse_loss = maskedMSE(fut_pred[:, :, 0, :], fut[:, :, 0, :], mask, 2)

        avg_loss += mse_loss.detach()
    avg_loss = avg_loss.item()

    print('Validation loss:', format(avg_loss / len_val, '0.4f'))

    info = {'val_loss': avg_loss / len_val}

    for tag, value in info.items():
        logger.add_scalar(tag, value, (epoch_num+1)*len_tr)
