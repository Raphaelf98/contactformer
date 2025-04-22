import argparse
import sys
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
import tqdm
# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTACT_FORMER_DIR = os.path.dirname(CONTACT_DIR)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
CONFIG_DIR = os.path.join(CONTACT_FORMER_DIR, 'config')

sys.path.append(os.path.join(BASE_DIR))

import config_parser
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import DataLoader
from contact_grasp_net.acronym_dataset import AcronymDataset
import importlib.util



from contact_grasp_net.conatact_graspnet_loss import ContacGraspNetLoss
from contact_grasp_net.checkpoint_io import CheckpointIO
from contact_grasp_net import utils

import wandb 
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"

def train(ContactGraspNet, global_config, log_dir, FLAGS):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = global_config['OPTIMIZER']['batch_size']
    if not FLAGS.debug:
        
        train_dataset = AcronymDataset(global_config=global_config,debug=False, device=device, train=True)
        test_dataset  = AcronymDataset(global_config=global_config,debug=False, device=device, train=False)
        
        wandb.init( project="contact grasp net", config={ 
        "optimizer": global_config['SCHEDULER']['optimizer']['type'],
        "learning_rate": global_config['SCHEDULER']['optimizer']['lr'],
        "architecture": "cgn-ptv3backbone",
        "dataset": "ACRONYM",
        "batch_size": global_config['OPTIMIZER']['batch_size'],
        "epochs": global_config['SCHEDULER']['epoch'],
        })

    else: 
        train_dataset = AcronymDataset(global_config=global_config,debug=True, device=device, train=True)
        test_dataset = AcronymDataset(global_config=global_config,debug=True, device=device, train=False)
        

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    grasp_net =  ContactGraspNet(global_config, device).to(device)
    loss_fcn = ContacGraspNetLoss(global_config, device).to(device)
    optimizer_type = global_config['OPTIMIZER']['optimizer']
    optimizer_params = global_config['SCHEDULER']['optimizer']
    if optimizer_type== 'adam':
        print("Using Adam optimizer")
        optimizer = torch.optim.Adam(grasp_net.parameters(),lr=global_config['OPTIMIZER']['learning_rate'])
    elif optimizer_type == 'adamw':
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(grasp_net.parameters(), lr=optimizer_params['lr'], weight_decay=optimizer_params['weight_decay'])
        scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=0.9,total_steps=global_config['SCHEDULER']['epoch'] * len(train_loader))
       
    logger = SummaryWriter(os.path.join(log_dir, 'logs'))
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(checkpoint_dir, model=grasp_net, opt=optimizer)

    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    
        
    cur_epoch = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)
    metric_val_best = load_dict.get('loss_val_best', np.inf)
    print_every = global_config['OPTIMIZER']['print_every'] \
        if 'print_every' in global_config['OPTIMIZER'] else 0
    checkpoint_every = global_config['OPTIMIZER']['checkpoint_every'] \
        if 'checkpoint_every' in global_config['OPTIMIZER'] else 0
    backup_every = global_config['OPTIMIZER']['backup_every'] \
        if 'backup_every' in global_config['OPTIMIZER'] else 0
    val_every = global_config['OPTIMIZER']['val_every'] \
        if 'val_every' in global_config['OPTIMIZER'] else 0
    max_epoch = global_config['OPTIMIZER']['max_epoch']

    
    # -------TRAINING LOOP--------
    
    for epoch_it in range(cur_epoch, max_epoch):
        grasp_net.train()
        pbar = tqdm.tqdm(train_loader)
        for i, data in enumerate(pbar):
            utils.send_dict_to_device(data, device)
            pc_cam = data['pc_cam']
            prediction = grasp_net(pc_cam)
            # for key, value in prediction.items():
            #     print(f"{key}: {value.shape}")
            loss, loss_info = loss_fcn(prediction, data)
            #Resets gradients of all model parameters to zero, 
            # gradients accumulate by default in Pytorch during backprop, 
            # clearing ensures update is only on current batch
            optimizer.zero_grad()
            # performs backpropagation to compute gradients of loss with respect to model parameters
            loss.backward()
            # updates model paramters based on computed gradients
            optimizer.step()
            #---------LOGGING----------
            for k, v in loss_info.items():
                logger.add_scalar(f'train/{k}', v, it)
            logger.add_scalar('train/loss', loss.item(), it)

            if not FLAGS.debug:
                if checkpoint_every and it % checkpoint_every == 0:
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                    loss_val_best=metric_val_best)   
                wandb.log({"loss": loss.item()})
                wandb.log(loss_info, commit=False)
            pbar.set_postfix({'loss': loss.item(),
                              'epoch': epoch_it})
            it += 1
            #--------VALIDATION ON TEST DATA------
        if val_every and epoch_it % val_every == 0:
            print("Running validation...")
            grasp_net.eval()
            with torch.no_grad():
                loss_log = []
                for val_it, data in enumerate(tqdm.tqdm(test_loader)):
                    utils.send_dict_to_device(data, device)
                    # Target contains input and target values
                    pc_cam = data['pc_cam']
                    pred = grasp_net(pc_cam)
                    loss, loss_info = loss_fcn(pred, data)
                    loss_log.append(loss.item())
                val_loss = np.mean(loss_log)
                logger.add_scalar('val/val_loss', val_loss, it)
                if not FLAGS.debug:
                    wandb.log({"validation/loss": val_loss}, step=it)
        if val_loss < metric_val_best:
            metric_val_best = val_loss
            if not FLAGS.debug:
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                    loss_val_best=metric_val_best)
        if optimizer_type == 'adamw':   
            scheduler.step()
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite_ckpt_dir', type=int, required=True, help='0, 1') # applies changes in contact_grasp_dir to files in checkpoint_dir. Attention: This will overwrite files in checkpoint_dir
    parser.add_argument('--model', type=str, required=True, help='ptv2, ptv3')
    parser.add_argument('--debug', type=int, default=0, help='0, 1')
    parser.add_argument('--config_dir', type=str, default=None, help='Config dir')
    parser.add_argument('--ckpt_dir', type=str, default=None,required=True, help='Checkpoint dir')

    FLAGS = parser.parse_args()
    checkpoint_dir = FLAGS.ckpt_dir
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Please check your installation.")

    CONFIG_FILE = FLAGS.config_file or "transformer_config.yaml"
    transformer_config_path = os.path.join(CONFIG_DIR,CONFIG_FILE)

    model_file_path = os.path.join(CONTACT_DIR, "conatact_graspnet_model.py")
    print("overwrite_ckpt_dir", FLAGS.overwrite_ckpt_dir)
    if FLAGS.overwrite_ckpt_dir: 
        config_parser.force_copy_file(source_file=transformer_config_path, target_directory=checkpoint_dir)
        config_parser.force_copy_file(source_file=model_file_path, target_directory=checkpoint_dir)
    else:
        config_parser.copy_file_if_not_exists(source_file=transformer_config_path, target_directory=checkpoint_dir)
        config_parser.copy_file_if_not_exists(source_file=model_file_path, target_directory=checkpoint_dir)

    model_file_path = os.path.join(checkpoint_dir, 'conatact_graspnet_model.py')
    if os.path.exists(model_file_path):
        spec = importlib.util.spec_from_file_location("checkpoiont_model", model_file_path)
        conatact_graspnet_model = importlib.util.module_from_spec(spec)
        sys.modules["checkpoiont_model"] = conatact_graspnet_model
        spec.loader.exec_module(conatact_graspnet_model)
        if FLAGS.model == 'ptv2':
            ContactGraspNet = conatact_graspnet_model.ContactGraspNetPtV2
            print("Using ContactGraspNetPtV2")
        elif FLAGS.model == 'ptv3':
            ContactGraspNet = conatact_graspnet_model.ContactGraspNetPtV3
            print("Using ContactGraspNetPtV3")
    # global_config = config_parser.load_config(FLAGS.config_dir)
    global_config = config_parser.load_config(config_path=os.path.join(checkpoint_dir, CONFIG_FILE))
    
    train(ContactGraspNet, global_config, checkpoint_dir, FLAGS)
