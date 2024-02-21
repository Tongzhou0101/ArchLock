# Import the search space
from naslib.search_spaces import NasBench201SearchSpace
import json
import logging
import os
import numpy as np
import torch

# import the Trainer used to run the optimizer on a given search space
from naslib.defaults.my_trainer_bp_emd import MyTrainer_BP_emd
# import the optimizers
from naslib.optimizers import (
    RandomSearch,
    MyRegularizedEvolution_BP_emd
)
# import the search spaces
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib import utils
from naslib.utils import get_dataset_api
from naslib.utils.log import setup_logger
from secure_nas.utils import *
from secure_nas.binary_predictor import BinaryRankPredictor
from fvcore.common.config import CfgNode # Required to read the config
###### End of imports ######


def update_config(config, optimizer_type, search_space_type):
    # Dataset being used
    # config.dataset_src = dataset_src
    # config.dataset_tar = dataset_tar
    # Directory to which the results/logs will be saved
    config.save = f"runs/{optimizer_type.__name__}/{search_space_type.__name__}/{config.dataset_src}/{config.dataset_tar}_{config.search.seed}"
    
    # # Seed used during search phase of the optimizer
    # config.search.seed = seed
    
def run_optimizer(pred, emd_list, optimizer_type, search_space_type,  config):
    # Update the config
    # update_config(config, optimizer_type, search_space_type)

    # Make the results directories
    os.makedirs(config.save + '/search', exist_ok=True)
    os.makedirs(config.save + '/eval', exist_ok=True)


     # See the config
    logger.info(f'Configuration is \n{config}')

    # Set up the seed
    # utils.set_seed(config.search.seed)

    # Instantiate the search space
    # n_classes = {
    #     'cifar10': 10,
    #     'cifar100': 100,
    #     'ImageNet16-120': 120
    # }
    # search_space_src = search_space_type(n_classes=n_classes[config.dataset_src])
    # search_space_tar = search_space_type(n_classes=n_classes[config.dataset_tar])
    search_space_src = search_space_type()
    search_space_tar = search_space_type()

    # Get the benchmark API
    logger.info('Loading Benchmark API')
    dataset_api_src = get_dataset_api(search_space_src.get_type(), config.dataset_src)
    dataset_api_tar = get_dataset_api(search_space_tar.get_type(), config.dataset_tar)
    
    # Instantiate the optimizer and adapat the search space to the optimizer
    optimizer = optimizer_type(config)
    optimizer.adapt_search_space(search_space_src, search_space_tar, dataset_api_src=dataset_api_src, dataset_api_tar=dataset_api_tar)

    # Create a Trainer
    trainer = MyTrainer_BP_emd(optimizer, config)

    # Perform the search
    trainer.search(pred, emd_list, report_incumbent=False)

    # Get the results of the search
    search_trajectory = trainer.search_trajectory
    print('Train accuracies src:', search_trajectory.train_acc_src)
    print('Validation accuracies src:', search_trajectory.valid_acc_src)
    print('Train accuracies tar:', search_trajectory.train_acc_tar)
    print('Validation accuracies tar:', search_trajectory.valid_acc_tar)

    # Get the validation performance of the best model found in the search phase
    best_model_val_acc_src, best_model_val_acc_tar = trainer.evaluate(dataset_api_src=dataset_api_src, dataset_api_tar=dataset_api_tar, metric=Metric.VAL_ACCURACY)
     
    
    # best_model_val_acc

    best_model = optimizer.get_final_architecture()

    return search_trajectory, best_model, best_model_val_acc_src, best_model_val_acc_tar




if __name__ == '__main__':

    # The configuration used by the Trainer and Optimizer
    # The missing information will be populated inside run_optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dset_list =  ['cifar10', 'cifar100', 'ImageNet16-120']

    
    config = {
        'dataset_src': 'ImageNet16-120',
        'dataset_tar': 'cifar100',
        'search': {
            'seed': 6006, # 
            'epochs': 40, # Number of epochs (steps) of the optimizer to run
            'fidelity': -1, # 
            'checkpoint_freq': 10,
            'acc_constraint': 45, # remember to change with src
            # Required by RegularizedEvolution
            'sample_size': 10,
            'population_size': 10, 
            'history_size': 30, 
            'emd': 15,
            'cos_sim': 0.9,
        },
        'pred':{
            'cfg': [128, 64, 32, 16],
            'path':'secure_nas/trained_net/nasbench201_cifar10_bp_emd.pt',
        },
        'save': 'runs' # folder to save the results to 
        # 'device': device
    }
       

    config = CfgNode.load_cfg(json.dumps(config))
    
 

    # Make the directories required for search and evaluation
    # os.makedirs(config['save'] + '/search', exist_ok=True)
    # os.makedirs(config['save'] + '/eval', exist_ok=True)

    # # Set up the loggers
    # logger = setup_logger()
    # logger.setLevel(logging.INFO)

    # See the config
    # logger.info(f'Configuration is \n{config}')
    # logger.info(config)

  

    # Instantiate the search space and get its benchmark API
    search_space = NasBench201SearchSpace()
    dataset_api = get_dataset_api('nasbench201', 'cifar10')
    # Set the optimizer and search space types
    # They will be instantiated inside run_optimizer
    optimizer_type = MyRegularizedEvolution_BP_emd # 
    search_space_type = NasBench201SearchSpace # {NasBench101SearchSpace, NasBench201SearchSpace, NasBench301SearchSpace}

    # Set the dataset
    # dataset_src = 'cifar10' # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    # dataset_tar = 'cifar100'
    
    config.save = f"runs/{optimizer_type.__name__}/{search_space_type.__name__}_{config.search.emd}_v1/"
    # Set up the loggers
    logger = setup_logger(config.save+'/search.log')
    logger.setLevel(logging.INFO)

    # Use a predictor with task embedding
    pred = BinaryRankPredictor(config.pred.cfg, in_ch=60+2048)   
    pred.to(device)
    pred.load_state_dict(torch.load(config.pred.path))

    emd_s = np.load('secure_nas/res/cf10_embedding.npy')[0]    
    emd_list = [emd_s.tolist()]
    for i in range(config['search']['emd']):
        ti = generate_embedding(emd_s, config.search.cos_sim).tolist()
        emd_list.append(ti) 

    acc_dict = {'cifar10': 90, 'cifar100': 71, 'ImageNet16-120': 45}
    dset_list_s = ['cifar100']
    for ts in dset_list_s:
        config.dataset_src = ts
        config.search.acc_constraint = acc_dict[ts]
        for tt in dset_list:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            config.dataset_tar = tt
            if ts == tt:
                continue

            search_trajectory, best_model, best_model_val_acc_src, best_model_val_acc_tar = run_optimizer(
                                                                    pred, emd_list,
                                                                    optimizer_type,
                                                                    search_space_type,
                                                                    config
                                                                )
            
