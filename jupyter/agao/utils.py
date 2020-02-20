
def get_n_params(model):
    n = 0
    for p in list(model.parameters()):
        n += p.nelement()
    return n

def print_progress(epoch, batch_idx, data, train_loader, loss):
    training_loss_list.append(loss.item())
    if batch_idx % 1 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

  