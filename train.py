import os
import torch
from utils import *

#Cross Validation
from sklearn.model_selection import KFold
#Save log
from torch.utils.tensorboard import SummaryWriter

# main
def train(dataloader, model, loss, device, num_epochs, save_path, n_folds = 5):
    results = {}
    folds = KFold(n_splits = n_folds)

    fff = []
    for i in range(n_folds):
        fff.append("fold"+str(i+1))
    workspace = save_path

    # KFold Cross Validation
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(dataloader)):
        if not os.path.isdir(os.path.join(workspace, fff[fold_])): os.makedirs(os.path.join(workspace, fff[fold_]))
        save_imgpath = os.path.join(workspace, fff[fold_])
        if not os.path.isdir(save_imgpath): os.makedirs(save_imgpath)
        log_dir = os.path.join(workspace, fff[fold_], 'log')
        if not os.path.isdir(log_dir): os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        print("fold n°{}".format(fold_+1))
        ##Split by folder and load by dataLoader
        train_subsampler = torch.utils.data.SubsetRandomSampler(trn_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=4, sampler=train_subsampler)
        valid_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=4, sampler=valid_subsampler)

        # Initialize Model
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        early_stopping = EarlyStopping(patience = 10, verbose = True, path=os.path.join(workspace, fff[fold_],'checkpoint.pt'))

        # Model selection
        best_iou = 0.0
        for epoch in range(num_epochs):        
            model.train()
            train_loss = []
            train_iou  = [] 
            valid_loss = []
            valid_iou  = []    
            cost_tmp = 0.0
            iou_tmp = 0.0       
            for batch_idx, (features,targets,file_name) in enumerate(train_dataloader):
                features = features.to(device)
                targets  = targets.to(device)        
                optimizer.zero_grad()

                ### FORWARD AND BACK PROP
                logits = model(features)
                cost = loss(logits, targets)            
                cost.backward()
                iou = iou_score(targets,logits).item()*100

                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                ### LOGGING
                train_loss.append(cost.item())
                train_iou.append(iou)

                cost_tmp += cost.item()
                iou_tmp += iou

                train_logits = logits.detach().cpu().numpy()
                if not batch_idx % 80:
                    print ('Epoch: %03d/%03d | Batch %03d/%03d | Train Loss: %.4f | Train IoU: %.4f%% '  
                           %(epoch+1, num_epochs, batch_idx, 
                             len(train_dataloader),
                             np.mean(train_loss),
                             np.mean(train_iou))
                          )
            writer.add_scalar('/train/loss', (cost_tmp/len(train_dataloader)), global_step=epoch)
            writer.add_scalar('/train/iou', (iou_tmp/len(train_dataloader)), global_step=epoch)

            ##Valid
            model.eval()
            cost_val_tmp = 0.0
            iou_val_tmp = 0.0
            with torch.no_grad():
                for batch_idx, (features,targets,file_name_val) in enumerate(valid_dataloader):

                    features = features.to(device)
                    targets  = targets.to(device)  

                    logits = model(features)
                    cost   = loss(logits, targets)
                    iou    = iou_score(targets,logits).item()*100

                    ### LOGGING
                    valid_loss.append(cost.item())
                    valid_iou.append(iou)
                print('Epoch: %03d/%03d |  Valid Loss: %.4f | Valid IoU: %.4f%%' % (
                      epoch+1, num_epochs, 
                      np.mean(valid_loss),
                      np.mean(valid_iou)))

                cost_val_tmp += cost.item()
                iou_val_tmp += iou
                valid_logits = logits.detach().cpu().numpy()

            writer.add_scalar('/valid/loss', (cost_val_tmp/len(valid_dataloader)), global_step=epoch)
            writer.add_scalar('/valid/iou', (iou_val_tmp/len(valid_dataloader)), global_step=epoch)

            # model save
            if best_iou <= np.mean(valid_iou):
                best_iou = np.mean(valid_iou)
                torch.save(model.state_dict(), os.path.join(workspace, fff[fold_]) + "/epoch%03d_model.pth" % epoch)

                # to save image
                np.save(os.path.join(save_imgpath, str(epoch)+'_train_file_name.npy'), file_name)
                np.save(os.path.join(save_imgpath, str(epoch)+'_train_output.npy'), train_logits)
                np.save(os.path.join(save_imgpath, str(epoch)+'_valid_file_name.npy'), file_name_val)
                np.save(os.path.join(save_imgpath, str(epoch)+'_valid_output.npy'), valid_logits)

            early_stopping(np.mean(valid_loss), model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            lr_scheduler.step()

        results[fold_+1] = best_iou

    # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {n_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

