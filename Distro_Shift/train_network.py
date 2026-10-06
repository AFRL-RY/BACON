## train_network.py

import json
import torch
import torch.nn as nn

class ActivationHookManager:
    def __init__(self):
        self.activations = {}
        self._handles = []

    def hook_fn(self, name):
        def hook(module, layer_input, layer_output):
            self.activations[name] = layer_output.detach()
        return hook

    def register(self, module, name):
        handle = module.register_forward_hook(self.hook_fn(name))
        self._handles.append(handle)

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.activations.clear()

def train_network(model, train_loader, dev_loader_train, hyperparameter_file, target_acc, device='cpu', save_best=False, root=None, save_file=None):

    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
    
    print("train_network.py parameters loading")
    lr_conv = float(jsonObject['lr_conv'])
    lr_class = float(jsonObject['lr_class'])
    decay_rate = float(jsonObject['decay_rate'])
    num_epochs = int(jsonObject['num_epochs'])
    train_batch_size = int(jsonObject['train_batch_size'])
    
    best_loss = float('inf')
    
    if save_best and root is not None:
        best_loss_filename = root + 'best_loss_acc.txt'
        with open(best_loss_filename, "w") as f:
            f.write("epoch,best_train_loss,best_dev_loss,accuracy\n")
            
    hook_manager = ActivationHookManager()
    
    try:
        hook_manager.register(model.ptm.classifier.Fc2, 'fc2')
        hook_manager.register(model.ptm.classifier.Fc3, 'fc3')
        print("Successfully registered hooks on VGG-16 fully-connected submodules.")
    except (AttributeError, IndexError):
        print("Warning: Model structure differs from standard VGG-16. Hooks not attached.")

    conv_params = [p for name, p in model.named_parameters() if "features" in name and p.requires_grad]
    classifier_params = [p for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': conv_params, 'lr': lr_conv},        
        {'params': classifier_params, 'lr': lr_class}   
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    for epoch in range(num_epochs):
        # --- TRAINING LOOP ---
        model.train()
        running_loss_t = 0.0
        
        for i, (images_t, labels_t) in enumerate(train_loader):
            optimizer.zero_grad()
            images_t = images_t.to(device)
            labels_t = labels_t.to(device)
            
            outputs_t = model(images_t)
            loss_t = criterion(outputs_t, labels_t)
            
            loss_t.backward()
            optimizer.step()
            running_loss_t += loss_t.item()

        epoch_loss_t = running_loss_t / len(train_loader)

-
        model.eval()  
        n_correct = 0
        n_samples = 0
        running_loss_d = 0.0
        
        with torch.no_grad():
            for images_d, labels_d in dev_loader_train:
                images_d = images_d.to(device)
                labels_d = labels_d.to(device)
                
                outputs_d = model(images_d)
                loss_d = criterion(outputs_d, labels_d)
                running_loss_d += loss_d.item()
                
                _, predicted = torch.max(outputs_d, 1)
                n_samples += labels_d.size(0)
                n_correct += (predicted == labels_d).sum().item()
        
        epoch_loss_d = running_loss_d / len(dev_loader_train)
        acc = 100.0 * n_correct / n_samples
        
        print(f"Epoch: {epoch+1} | train loss: {epoch_loss_t:.4f} | dev loss: {epoch_loss_d:.4f} | dev acc: {acc:.2f}%")
        
        if epoch + 1 == 2:
            batch_preds = torch.max(outputs_t, 1)[1].cpu()
            preds_div = batch_preds.divide(batch_preds[0]).sum()
            is_bad = preds_div.equal(torch.tensor(train_batch_size, dtype=torch.float32))
            if is_bad:
                print('bad start')
                return model, True

        scheduler.step(epoch_loss_d)
        -
        if save_best and epoch_loss_d <= best_loss:
            best_loss = epoch_loss_d  
            if root is not None:
                torch.save(model.state_dict(), root + 'best_model.pth')
                
                with open(best_loss_filename, "a") as best_loss_acc:
                    best_loss_acc.write(f"{epoch+1},{epoch_loss_t:.4f},{epoch_loss_d:.4f},{acc:.2f}\n")
        
        if acc >= target_acc:
            print(f"Target accuracy of {target_acc}% met. Stopping training.")
            break

    print('Finished Training\n')
    
    hook_manager.remove_hooks()
    print("Hooks successfully removed. Memory cleared.")
    
    if root is not None and save_best:
        model.load_state_dict(torch.load(root + 'best_model.pth'))
    return model

