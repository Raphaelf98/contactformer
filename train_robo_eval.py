import argparse
import sys
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
import tqdm
# Import pointnet library
CONTACT_FORMER_DIR = os.path.dirname(os.path.abspath(__file__))
CONTACT_DIR = os.path.join(CONTACT_FORMER_DIR, 'contact_grasp_net')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
THESIS_DIR = os.path.dirname(BASE_DIR)
THESIS_EXPERIMENTS_DIR = os.path.join(os.path.dirname(CONTACT_FORMER_DIR), 'thesis_experiments')
CONFIG_DIR = os.path.join(CONTACT_FORMER_DIR, 'config')
sys.path.append(os.path.join(BASE_DIR))

sys.path.append(THESIS_EXPERIMENTS_DIR)
from thesis_experiments.train_eval.train_eval import train_eval, virtual_train_eval

import contact_grasp_net.config_parser
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import DataLoader, RandomSampler
from contact_grasp_net.acronym_dataset import AcronymDataset
import importlib.util
from pyvirtualdisplay import Display

import multiprocessing as mp


from contact_grasp_net.conatact_graspnet_loss import ContacGraspNetLoss
from contact_grasp_net.checkpoint_io import CheckpointIO
from contact_grasp_net import utils

import wandb 
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"

def train(ContactGraspNet, global_config, log_dir, DEBUG):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = global_config['OPTIMIZER']['batch_size']
    seed = global_config['OPTIMIZER'].get('seed', 42)
    
    train_dataset = AcronymDataset(global_config=global_config,debug=DEBUG, device=device, train=True)
    test_dataset  = AcronymDataset(global_config=global_config,debug=DEBUG, device=device, train=False)
   
    train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(seed))
    test_sampler = RandomSampler(test_dataset, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size, sampler=test_sampler)
    
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
    print(f"Resume training at epoch: {cur_epoch}")
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
    robot_val_every = global_config['OPTIMIZER'].get('robot_val_every', 0)
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
            torch.cuda.empty_cache()

            #---------LOGGING----------
            for k, v in loss_info.items():
                logger.add_scalar(f'train/{k}', v, it)
            logger.add_scalar('train/loss', loss.item(), it)

            if not DEBUG:
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
                if not DEBUG:
                    wandb.log({"validation/loss": val_loss})

                if val_loss < metric_val_best:
                    metric_val_best = val_loss
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,loss_val_best=metric_val_best)
                
        
        if optimizer_type == 'adamw':   
            scheduler.step()

        if robot_val_every and epoch_it % robot_val_every == 0:
            print("Running robot validation...")
            epoch_it = epoch_it + 1
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
            return epoch_it, checkpoint_io, it, metric_val_best


def evaluate_model(best_ckpt_path, epoch_it,config_file,name=None):
    # 
    queue = mp.Queue()
    p = mp.Process(target=eval_worker, args=(epoch_it, best_ckpt_path,config_file, queue, name))
    p.start()
    p.join()
    grasp_success, object_grasp_ratio = queue.get()
    return  grasp_success, object_grasp_ratio

def eval_worker(epoch_it, ckpt_path, config_file,  queue, name=None):
    grasp_success, object_grasp_ratio = virtual_train_eval('ContactFormer', ckpt_path, epoch=epoch_it, CONFIG_FILE=config_file, NAME=name)
   
    queue.put((grasp_success, object_grasp_ratio))

    
if __name__=="__main__":
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        gpu_id = 0
        print("Total Memory:", torch.cuda.get_device_properties(gpu_id).total_memory / 1e6, "MB")
        print("Allocated Memory:", torch.cuda.memory_allocated(gpu_id) / 1e6, "MB")
        print("Cached Memory:", torch.cuda.memory_reserved(gpu_id) / 1e6, "MB")

    else:
        print("CUDA is not available. Please check your installation.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite_ckpt_dir', type=int, required=True, help='0, 1') # applies changes in contact_grasp_dir to files in checkpoint_dir. Attention: This will overwrite files in checkpoint_dir
    parser.add_argument('--model', type=str, required=True, help='ptv2, ptv3')
    parser.add_argument('--debug', type=int, default=0, help='0, 1')
    parser.add_argument('--config_file', type=str, default=None, help='Config dir')
    parser.add_argument('--ckpt_dir', type=str, default=None,required=True, help='Checkpoint dir')
    parser.add_argument('--resume', type=str, default=None, help='Provide run ID to resume training')

    FLAGS = parser.parse_args()
    checkpoint_dir = FLAGS.ckpt_dir
    DEBUG = FLAGS.debug
    CONFIG_FILE = FLAGS.config_file or "transformer_config.yaml"
    transformer_config_path = os.path.join(CONFIG_DIR,CONFIG_FILE)
    model_file_path = os.path.join(CONTACT_DIR, "conatact_graspnet_model.py")
    print("overwrite_ckpt_dir", FLAGS.overwrite_ckpt_dir)
    print("config_file", transformer_config_path)

    if FLAGS.overwrite_ckpt_dir: 
        contact_grasp_net.config_parser.force_copy_file(source_file=transformer_config_path, target_directory=checkpoint_dir)
        contact_grasp_net.config_parser.force_copy_file(source_file=model_file_path, target_directory=checkpoint_dir)
    else:
        contact_grasp_net.config_parser.copy_file_if_not_exists(source_file=transformer_config_path, target_directory=checkpoint_dir)
        contact_grasp_net.config_parser.copy_file_if_not_exists(source_file=model_file_path, target_directory=checkpoint_dir)

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
    global_config = contact_grasp_net.config_parser.load_config(config_path=os.path.join(checkpoint_dir,CONFIG_FILE))
    
    disp = Display(visible=0, size=(1024, 768))
    disp.start()
    if not DEBUG:
        if not FLAGS.resume:
            # Create a run ID using checkpoint name and current date
            checkpoint_name = os.path.basename(FLAGS.ckpt_dir)
            current_date = datetime.now().strftime("%Y%m%d%H%M%S")
            run_id = f"{checkpoint_name}_{current_date}"
        else:
            run_id = FLAGS.resume
       
        wandb.init( 
            project="ContactFormer", 
            resume="allow",
            id=run_id,
            config={ 
                "optimizer": global_config['SCHEDULER']['optimizer']['type'],
                "learning_rate": global_config['SCHEDULER']['optimizer']['lr'],
                "architecture": "cgn-ptv2backbone",
                "dataset": "ACRONYM",
                "batch_size": global_config['OPTIMIZER']['batch_size'],
                "epochs": global_config['SCHEDULER']['epoch'],
                "encoder": global_config['MODEL']['ENCODER'],
                "decoder": global_config['MODEL']['DECODER'],
                }
            )
    epoch_it = 0
    
    import time
    mp.set_start_method('spawn', force=True)
    current_best_grasp_success = 0
    while epoch_it < global_config['OPTIMIZER']['max_epoch']:
        torch.cuda.empty_cache()  
        time.sleep(5)
        epoch_it, checkpoint_io, it, metric_val_best = train(ContactGraspNet, global_config, FLAGS.ckpt_dir, DEBUG)
        torch.cuda.empty_cache()  
        time.sleep(5)
        print(f"Epoch {epoch_it} completed. Evaluating model...")
        
        ckpt_path = os.path.join(CONTACT_FORMER_DIR, FLAGS.ckpt_dir)
                

        grasp_success, object_grasp_ratio = evaluate_model(ckpt_path, epoch_it, CONFIG_FILE, FLAGS.model)
        if grasp_success > current_best_grasp_success:
            current_best_grasp_success = grasp_success
            print("Best grasp success so far:", current_best_grasp_success)
            checkpoint_io.save('model_best_val.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best, grasp_success=grasp_success)
        print("Grasp success:", grasp_success, "Object grasp ratio:", object_grasp_ratio)
        if not DEBUG:
            wandb.log({
                'grasp_success': grasp_success,
                'object_grasp_ratio': object_grasp_ratio
            })
                
    disp.stop()
