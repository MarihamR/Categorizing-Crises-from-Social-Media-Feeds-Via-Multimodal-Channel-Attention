##Files
from args import get_args
from trainer import Trainer
#from dmd_dataset import DMDDataset
from crisismmd_dataset import CrisisMMDataset
from models import MCAModel, ImageOnlyModel, TextOnlyModel
from focalloss import *

from os import path as osp
import os
import logging
import time
import random
from PIL.Image import SAVE
from torch.serialization import save
import numpy as np
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch import nn, optim 
import torchvision.transforms as transforms
import nltk
nltk.download('stopwords')

a=15
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(a)
np.random.seed(a)
torch.manual_seed(a)
torch.cuda.manual_seed(a)
os.environ['CURL_CA_BUNDLE'] = ''



if __name__ == '__main__':

    ## Loading Arguments
    opt = get_args()

    model_to_load = opt.model_to_load
    image_model_to_load = opt.image_model_to_load
    text_model_to_load = opt.text_model_to_load

    device = opt.device
    num_workers = opt.num_workers

    EVAL = opt.eval
    USE_TENSORBOARD = opt.use_tensorboard
    SAVE_DIR = opt.save_dir
    MODEL_NAME = opt.model_name if opt.model_name else str(int(time.time()))
    pred_file=opt.pred_file

    MODE = opt.mode
    TASK = opt.task
    MAX_ITER = opt.max_iter
    OUTPUT_SIZE = None 
    if TASK == 'task1':
        OUTPUT_SIZE = 2
    elif TASK == 'task2':
        OUTPUT_SIZE = 8
    elif TASK == 'task3':
        OUTPUT_SIZE = 3
    elif TASK == 'DMD':
        OUTPUT_SIZE = 6        
    else:
        raise NotImplemented


    # General hyper parameters
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size

    # Create folder for saving
    save_dir = osp.join(SAVE_DIR, MODEL_NAME)
    if pred_file!=None:
        pred_file=  osp.join(save_dir, pred_file)
    
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    # set logger
    logging.basicConfig(filename=osp.join(save_dir, 'output_{}.log'.format(int(time.time()))), level=logging.INFO)

########################################### Datasests and Dataloaders#####################################
    train_loader, dev_loader = None, None
    if not EVAL: # if training
        train_set = CrisisMMDataset()
        train_set.initialize(opt, phase='train', cat='all',task=TASK)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dev_set = CrisisMMDataset()
    dev_set.initialize(opt, phase='dev', cat='all',task=TASK)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_set = CrisisMMDataset()
    test_set.initialize(opt, phase='test', cat='all',task=TASK)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
########################################### Loss Function #############################################
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn=FocalLoss(gamma=0.5)
############################################ Models####################################################    
    if MODE == 'text_only':
        model = TextOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)

    elif MODE == 'image_only':
        model = ImageOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
        
    elif MODE == 'both':
        model =  MCAModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    
    else:
        raise NotImplemented
############################################### optimizer and scheduler #################################################### 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=4, cooldown=0, verbose=True)
##################################################Training and testing #####################################################
    trainer = Trainer(train_loader, dev_loader, test_loader,
                      model, loss_fn, optimizer, scheduler, eval=EVAL, device=device, tensorboard=USE_TENSORBOARD, mode=MODE,pred_file=pred_file)

    if model_to_load:
        #model.load(model_to_load)
        model.load_state_dict(torch.load(model_to_load))
        logging.info("\n***********************")
        logging.info("Model Loaded!")
        logging.info("***********************\n")
    if text_model_to_load:
        #model.load(text_model_to_load)
        model.load_state_dict(torch.load(text_model_to_load))
    if image_model_to_load:
        #model.load(image_model_to_load)
        model.load_state_dict(torch.load(image_model_to_load))

    if not EVAL:
        logging.info("\n================Training Summary=================")
        logging.info("Training Summary: ")
        logging.info("Learning rate {}".format(learning_rate))
        logging.info("Batch size {}".format(batch_size))
        logging.info(trainer.model)
        logging.info("\n=================================================")

        trainer.train(MAX_ITER)

    else:
        logging.info("\n================Evaluating Model=================")
        logging.info(trainer.model)

        trainer.validate()
        trainer.predict()
