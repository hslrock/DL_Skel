import torch
def infer(config,valid_dl,model):
    '''
    config.optimizer: Training optimizer eg) Adam,SGD....
    config.criterion: Loss eg) CrossEntropy, 
    config.lr: learning_rate
    train_dl,valid_dl: Dataloader
    model: target model
    save_path
    '''
    model.to(config["device"])

    model.eval()

    valid_loss=0

    correct=0
    total=0

    for batch_idx,(img,label) in enumerate(valid_dl):
        img=img.to(config["device"])
        label=label.to(config["device"])
            
        output=model(img)
        loss=config["crit"](output,label)
        #Update Parameters
        valid_loss+=loss.item()

        if config["accuracy"]:
            _, predicted = torch.max(output.data, 1)
            total+=label.size()[0]
            correct += (predicted == label).sum().item()
    avg_valid_loss=valid_loss/(len(valid_dl))
  #  config["valid_log"].append(avg_valid_loss)
   # config["valid_acc_log"].append(correct/total)
   # print()
    print(f'Validation_Loss: {avg_valid_loss}')
    print(f'Validation_Acc: {correct/total}')

    return config,model
            
    