import math
import torch
from torch.optim import Optimizer
from datamanagement import load # not an open source yet
import adabound
import eval_net # not an open source yet
import model_new # not an open source yet
from dw import eval as dwe

path_checkpoint = './same folder where you have your script/' 

def train_net(net,
              epochs=2,
              batch_size=1,
              lr=0.001,
              device):
    
    #torch.cuda.synchronize()
    #load_start = timer()
    SD_tra_dataset = load('c2',subset='train',eval=False)
    SD_train_loader = DataLoader(SD_tra_dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                              pin_memory=True,
                             drop_last=True)
    TD_tra_dataset = load('s',subset='all',eval=False)
    TD_train_loader = DataLoader(TD_tra_dataset, batch_size=batch_size, shuffle=False, num_workers=1, 
                                 pin_memory=True,
                                 drop_last=True)
 
    val_dataset = load('s',subset='test',eval=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, 
                            drop_last=False)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Device:          {device.type}
    ''')
    
    optimizer = adabound.AdaBound(net.parameters(), lr=lr, final_lr=0.1) #AdaBound
  
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train() #net.val() should be called for each batch not just each epoch
        
        epoch_loss = 0
  
        n_batch = len(train_loader) # [images (patches) per epoche]/batch_size
        
        train_CM = np.zeros((6,6),dtype=np.long)
        
        with tqdm(total=n_batch, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            
           
            for TD_batch in TD_train_loader:
                pbar.update(batch_size)
    
                """training in source doamin"""
                for SD_batch in SD_train_loader: 
                    SD_true_masks = SD_batch['labels'] 
                    imgs = SD_batch['image']
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    SD_true_masks = SD_true_masks.to(device=device, dtype=mask_type)
                    break

                SD_logits,SD_probs,SD_masks_pred = net(imgs) 
                SD_loss = criterion(SD_logits, SD_true_masks) 
         
                """training in target domain"""
                TD_imgs = TD_batch['image']
                imgs_temp = TD_imgs.cpu().numpy()
                TD_imgs = TD_imgs.to(device=device, dtype=torch.float32)
                
                TD_labels_cpu = TD_batch['labels']
                TD_labels_cpu_np = TD_labels_cpu.data.numpy()
                
                TD_logits,TD_probs,pseudo_labels = net(TD_imgs)
                pseudo_labels_cpu_np = pseudo_labels.cpu().numpy()
                
                TD_loss = criterion(TD_logits,pseudo_labels)
                epoch_TD_loss += TD_loss.item()/n_batch
                
                dwe.update_confusion_matrix(train_CM, 
                                            pseudo_labels_cpu_np, 
                                            TD_labels_cpu_np)
            
                #net_forward_time += net_forward_end
                
                """joint training and the total loss"""
                total_loss = (SD_loss+ TD_loss)/2 
                #if the two losses agree with each other, coresponding to increse gradient/double learning rate 

                pbar.set_postfix(**{'loss (batch)': total_loss.item()})

                optimizer.zero_grad()
                
                total_loss.backward()
                
                optimizer.step()
                torch.cuda.empty_cache()# release memory 
        
        val_f1s,val_ious,val_oa = eval_net(val_dataset, val_loader, net, device) #test_set,test_loader, net, device
        
        torch.cuda.empty_cache()

        train_metrics = dwe.get_confusion_metrics(train_CM)
        f1s = train_metrics['f1s']
        ious = train_metrics['ious']
        oa = train_metrics['oa']
        
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            'epoch_loss': total_loss,
            'train_accuracy':oa,
            'train_f1s':f1s,
            'train_ious':ious,
            
            'val_accuracy':val_oa,
            'val_f1s':val_f1s,
            'val_ious':val_ious
            }, path_checkpoint + f"CP_epoch{epoch + 1}.pth")
        logging.info(f'Checkpoint {epoch + 1} saved !')