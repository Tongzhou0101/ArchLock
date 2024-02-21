import collections
import logging
import torch
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph

from naslib.utils import count_parameters_in_MB
from naslib.utils.log import log_every_n_seconds

from fvcore.common.config import CfgNode
from secure_nas.utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
logger = logging.getLogger(__name__)
# logger.propagate = False

class MyRegularizedEvolution_BP_emd(MetaOptimizer):
    # used in MyTrainer
    using_step_function = False

    def __init__(self, config: CfgNode):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size
        self.history_size = config.search.history_size

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset_src = config.dataset_src
        self.dataset_tar = config.dataset_tar

        self.acc_constraint = config.search.acc_constraint

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()




    def adapt_search_space(self, search_space_src: Graph, search_space_tar: Graph, scope: str = None, dataset_api_src: dict = None, dataset_api_tar: dict = None):
        assert (
            search_space_src.QUERYABLE and search_space_tar.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space_src = search_space_src.clone()
        self.search_space_tar = search_space_tar.clone()
        self.scope = scope if scope else search_space_src.OPTIMIZER_SCOPE
        self.dataset_api_src = dataset_api_src
        self.dataset_api_tar = dataset_api_tar

    def new_epoch(self, epoch: int, pred, emd_list):
        # We sample as many architectures as we need
        # This function is used to update population pool
        if epoch < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            # If there is no scope defined, let's use the search space default one

            model = (
                torch.nn.Module()
            )
            
            
            acc_flag = True # used to determine if has an accuracy constraint
            
            while acc_flag:
                model.arch = self.search_space_src.clone()

                model.arch.sample_random_architecture(dataset_api=self.dataset_api_src)
                
                #  queried acc can be replaced with the predicted score in pratice 
                model.accuracy_src = model.arch.query(
                    self.performance_metric, self.dataset_src, dataset_api=self.dataset_api_src
                )

                if model.accuracy_src < self.acc_constraint:
                    # print(f'Model acc {model.accuracy_src} < acc constraint {self.acc_constraint}. Generate a new candidate')
                    continue
                # model.accuracy_tar = model.arch.query(
                #     self.performance_metric, self.dataset_tar, dataset_api=self.dataset_api_tar
                # )

                
                # model.fitness = - self.weighted * model.accuracy_tar
                acc_flag = False



            self.population.append(model)
            self._update_history(model,pred, emd_list)
            log_every_n_seconds(
                logging.INFO, "Population size {}".format(len(self.population))
            )
        else:
            # sort the population pool each update for source (s) and each simulated target (ti)
            sorted_s, sorted_ti = sort_arch_emd(self.population, pred, emd_list, device)
            for i in range(len(self.population)):
                self.population[i].src_rank = sorted_s.index(i)
                self.population[i].tari_rank = sorted_ti.index(i)
                self.population[i].rank_diff = self.population[i].tari_rank - 2*self.population[i].src_rank
            
            # best = max(self.population, key=lambda x: x.rank_diff)
            
            
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            # randomly sample k archs in the current population and choose the best one to mutate
            parent = max(sample, key=lambda x: x.rank_diff)

            child = (
                torch.nn.Module()
            )
            


            acc_flag = True
            
            cnt = 0
            while acc_flag:
                
                child.arch = self.search_space_src.clone()
                if cnt > 15: 
                    child.arch.sample_random_architecture(dataset_api=self.dataset_api_src)
                else:
                    child.arch.mutate(parent.arch, dataset_api=self.dataset_api_src)
                child.accuracy_src = child.arch.query(
                    self.performance_metric, self.dataset_src, dataset_api=self.dataset_api_src
                )

                if child.accuracy_src < self.acc_constraint:
                    cnt += 1
                    # print(f'Child acc {child.accuracy_src} < acc constraint {self.acc_constraint}. Generate a new candidate')
                    continue
                
              
                # child.fitness = child.accuracy_src - self.weighted * child.accuracy_tar

                acc_flag = False


            # remove the oldest one
            self.population.popleft()
            self.population.append(child)
            self._update_history(child,pred, emd_list)
            

    
    def _update_history(self, sample,pred, emd_list):
        sample.accuracy_tar = sample.arch.query(
                self.performance_metric, self.dataset_tar, dataset_api=self.dataset_api_tar
            )
        self.history.append(sample)

        if len(self.history) > self.history_size:
            
            sorted_s, sorted_ti = sort_arch_emd(self.history, pred, emd_list, device)
            for i in range(len(self.history)):
                self.history[i].src_rank = sorted_s.index(i)
                self.history[i].tari_rank = sorted_ti.index(i)
                self.history[i].rank_diff = self.history[i].tari_rank - 2*self.history[i].src_rank
            
            self.history =  sorted(self.history, key=lambda x: x.rank_diff, reverse=True)[:self.history_size]

    def train_statistics(self, report_incumbent: bool = True):
        # # report_incumbent is set to True, thus returning the best one; otherwise return the newest onr
                
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.population[-1].arch

        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset_src, dataset_api=self.dataset_api_src
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset_src, dataset_api=self.dataset_api_src
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset_src, dataset_api=self.dataset_api_src
            ),
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset_tar, dataset_api=self.dataset_api_tar
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset_tar, dataset_api=self.dataset_api_tar
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset_tar, dataset_api=self.dataset_api_tar
            )
         
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_final_architecture(self):
        #seach for the best arch
        # if we use binary predictor, instead of using max function, we should use ranking algorthm to select the best one 
             
        return min(self.history, key=lambda x: x.accuracy_tar).arch
        # return max(self.history, key=lambda x: x.rank_diff).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)



