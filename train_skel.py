import torch
def train(config,train_dl,valid_dl,model):
    '''
    config.optimizer: Training optimizer eg) Adam,SGD....
    config.criterion: Loss eg) CrossEntropy, 
    config.lr: learning_rate
    train_dl,valid_dl: Dataloader
    model: target model
    save_path
    '''
    model.to(config["device"])
    
    for epoch in range(config["epochs"]):
        print(f'Epochs : {epoch+1}')
        model.train()
        avg_train_loss=0
        correct=0
        total=0
        for batch_idx,(img,label) in enumerate(train_dl):
            #Zero_grad_Optimizer
            config["optim"].zero_grad()
            img=img.to(config["device"])
            label=label.to(config["device"])
            output=model(img)
            
            #Measure Loss
            loss=config["crit"](output,label)
            loss.backward()
            
            if config["accuracy"]:
                _, predicted = torch.max(output.data, 1)
                total+=label.size()[0]
                correct += (predicted == label).sum().item()
            
            #Update Parameters
            config["optim"].step()
            avg_train_loss+=loss.item()
            if batch_idx % config["log_interval"]==0:
                config["train_log"].append(avg_train_loss/(batch_idx+1))
                config["train_acc_log"].append(correct/(total))
                print()
                print(f"Batch {batch_idx+1}/{len(train_dl)} Loss: {loss.item()}")
                print(f"Accuracy {correct/(total)}")
        model.eval()
        
        valid_loss=0
        
        correct=0
        total=0
        
        for batch_idx,(img_label) in enumerate(valid_dl):
            output=model(img)
            #Measure Loss
            loss=config["crit"](output,label)
            #Update Parameters
            valid_loss+=loss.item()
            if config["accuracy"]:
                _, predicted = torch.max(output.data, 1)
                total+=label.size()[0]
                correct += (predicted == label).sum().item()
        avg_valid_loss=valid_loss/(len(valid_dl))
        config["valid_log"].append(avg_valid_loss)
        config["valid_acc_log"].append(correct/total)
        print()
        print(f'Validation_Loss: {config["valid_log"][-1]}')
        print(f'Validation_Acc: {config["valid_acc_log"][-1]}')
        
        if min(config["valid_log"]) > avg_valid_loss:
            print()
            print('Validation Result is better, saving the new model')
            torch.save(model.state_dict(), config["save_dir"]+f"epoch_{epoch}")
        
        
        
    return config,model
            
    