# train_network.py

import json
import torch
import torch.nn as nn

def train_network(model, train_loader, dev_loader_train, hyperparameter_file, target_acc, device='cpu', save_best=False,root=None, save_file=None):
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    print("train_network.py parameters loading")
    learning_rate = float(jsonObject['learning_rate'])
    decay_rate = float(jsonObject['decay_rate'])
    num_epochs = int(jsonObject['num_epochs'])
    train_batch_size = int(jsonObject['train_batch_size'])


    print("learning_rate:",learning_rate)
    print("decay_rate:",decay_rate)
    print("num_epochs:",num_epochs)
    print("train_batch_size:",train_batch_size)
    
    best_loss = 999999999
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    # Register forward hook to be able to extract activations to compute theta
    # fc2 and fc3 are the last two fully connected layers
    model.register_forward_hook(get_activation('fc2'))  # used to compute phi
    model.register_forward_hook(get_activation('fc3'))  # Used in test to compute softmax
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = decay_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    n_total_steps = len(train_loader)

    model.train()
    for epoch in range(num_epochs):
        running_loss_t = 0.0
        for i, (images_t, labels_t) in enumerate(train_loader):
            optimizer.zero_grad()
            images_t = images_t.to(device)
            labels_t = labels_t.to(device)

            # Forward pass
            outputs_t = model(images_t)
            loss_t = criterion(outputs_t, labels_t)

            # Backward and optimize
            loss_t.backward()
            optimizer.step()

            running_loss_t += loss_t.item()
            
        with torch.no_grad():
                n_correct = 0
                n_samples = 0
                running_loss_d = 0.0
                for i, (images_d, labels_d) in enumerate(dev_loader_train):
                    images_d = images_d.to(device)
                    labels_d = labels_d.to(device)

                    # Forward pass
                    outputs_d = model(images_d)
                    loss_d = criterion(outputs_d, labels_d)
            
                    running_loss_d += loss_d.item()
                    _, predicted = torch.max(outputs_d,1)
                    n_samples += labels_d.size(0) 
                    n_correct += (predicted == labels_d).sum().item()
                    
                model.train()
        
        epoch_loss_t = running_loss_t / len(train_loader)
        running_loss_t = 0.0
        
        epoch_loss_d = running_loss_d / len(dev_loader_train)
        running_loss_d = 0.0

        print('Epoch:',epoch+1,' train loss: ',epoch_loss_t,' | dev loss: ',epoch_loss_d)
        


        if epoch+1 == 2:
            batch_preds = torch.max(outputs_t,1)[1].cpu()
            preds_div = batch_preds.divide(batch_preds[0]).sum()
            is_bad = preds_div.equal(torch.tensor(train_batch_size, dtype = torch.float32))
            if is_bad:
                print('bad start')
                return model, True
            
        scheduler.step(epoch_loss_d)
        
        #Add in compute of accuracy, add in saving best loss
        if save_best and epoch_loss_d <= best_loss:
            best_model = model.state_dict()
            best_loss = epoch_loss_d
            if root is not None:
                acc = 100.0 * n_correct/n_samples
                torch.save(model.state_dict(), root+'best_model.pth')
                best_loss_filename = root+'best_loss_acc.txt'
                best_loss_acc = open(best_loss_filename,"w")
                best_loss_acc.write("epoch,best_loss, accuracy \n")
                best_loss_acc.close()
                best_loss_acc = open(best_loss_filename,"a")
                best_loss_acc.write(str(epoch+1))
                best_loss_acc.write(',')
                best_loss_acc.write(str(best_loss))
                best_loss_acc.write(',')
                best_loss_acc.write(str(acc))
                best_loss_acc.close()        
        #Test if met accuracy reqmt, and exit if true
        if acc >= target_acc:
            break
        
        
    #writer.close()
    print('Finished Training\n')

        
    model.load_state_dict(torch.load(root+'best_model.pth'))
    
    return model
