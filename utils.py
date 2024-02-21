#%%
import torch
from naslib.utils import get_dataset_api
import numpy as np
from naslib.utils import get_dataset_api
from naslib.search_spaces.nasbench201.conversions import *




def sort_arch_emd(popu, pred, emd_list, device):
    
    # Create pairwise comparison matrix
    comp_matrix_s = torch.zeros(len(popu), len(popu), dtype=torch.int)
    comp_matrix_t = torch.zeros(len(popu), len(popu), dtype=torch.int)
    for i in range(len(popu)):
        for j in range(i + 1, len(popu)):
            arch1_encode = popu[i].arch.encode()
            arch2_encode = popu[j].arch.encode()
            # print('encode',arch1_encode)
            arch_pair = arch1_encode+arch2_encode
            pair_src = torch.tensor(arch_pair+emd_list[0]).float().to(device)
        
             
            # print('cuda',pair.dtype)
            if pred(pair_src): # pred will output 1 if i is greater than j
                comp_matrix_s[i, j] = 1
            else:
                comp_matrix_s[j, i] = 1
            for emd in emd_list[1:]:
                pair_ti = torch.tensor(arch_pair+emd).float().to(device)
                if pred(pair_ti): # pred will output 1 if i is greater than j
                    comp_matrix_t[i, j] += 1
                else:
                    comp_matrix_t[j, i] += 1

    C = 1
    adj_matrix_s = C - C * comp_matrix_s
    adj_matrix_t = C - C * comp_matrix_t
    # Sort the items using the topological sorting algorithm
    sorted_indices_s = torch.topk(adj_matrix_s.sum(dim=0), k=len(popu)).indices
    sorted_indices_t = torch.topk(adj_matrix_t.sum(dim=0), k=len(popu)).indices

    return sorted_indices_s.tolist(), sorted_indices_t.tolist()

def generate_embedding(source, similarity):
    # Normalize the source vector
    source_norm = np.linalg.norm(source)
    source_normalized = source / source_norm

    # Compute the angle between the source and target vectors
    theta = np.arccos(similarity)

    # Compute the length of the target vector
    length_target = similarity * source_norm

    # Generate a random orthogonal vector to the source vector
    ortho = np.random.randn(*source.shape)
    ortho -= np.dot(ortho, source_normalized) * source_normalized
    ortho_norm = np.linalg.norm(ortho)
    ortho_normalized = ortho / ortho_norm

    # Compute the target vector using the angle and length
    target = np.cos(theta) * length_target * source_normalized + np.sin(theta) * length_target * ortho_normalized

    return target

def arch2res(search_space, dataset, arch_en):
    benchmark_api = get_dataset_api(search_space=search_space, dataset=dataset)

   
    arch_str = convert_op_indices_to_str(arch_en)
    img = benchmark_api["nb201_data"][arch_str]
    if dataset == 'cifar10':
        acc = img['cifar10-valid']["eval_acc1es"][-1]
    else:
        acc = img[dataset]["eval_acc1es"][-1]
    return acc