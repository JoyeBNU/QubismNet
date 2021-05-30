from DMRGalgorithms import dmrg_finite_size
import Parameters as Pm
import numpy as np
from os import path
from BasicFunctions import load_pr, save_pr, plot, output_txt, print_dict
from TensorBasicModule import open_mps_product_state_spin_up
import pickle

paras = np.arange(0.4, 1.0, 0.06)

lattice = 'kagome2n'
para_dmrg = Pm.generate_parameters_dmrg(lattice)
para_dmrg['spin'] = 'half'
para_dmrg['num_columns'] = 2
para_dmrg['bound_cond'] = 'pbc'
para_dmrg['chi'] = 16
para_dmrg['jxy'] = 1
para_dmrg['jz'] = 1
para_dmrg['hx'] = 0
para_dmrg['hz'] = 0
para_dmrg['project_path'] = '.'
para_dmrg['data_path'] = 'data/HeisenbergKagome2n'

info_list, ob_list = [], []
ob_e_persite_list = []
for i in range(len(paras)):
    para_dmrg['ratio'] = paras[i]
    para_dmrg = Pm.make_consistent_parameter_dmrg_2(para_dmrg)
    #print_dict(para_dmrg)
    ob, a, info, para = dmrg_finite_size(para_dmrg)
    ob_list.append(ob)
    info_list.append(info)
    ob_e_persite_list.append(ob['e_per_site'])
    #print('Energy per site = ' + str(ob['e_per_site']))
    #save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para), ('ob', 'a', 'info', 'para'))

# if path.isfile(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr')):
#     print('Load existing data ...')
#     a, ob = load_pr(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr'), ['a', 'ob'])
# else:
#     ob, a, info, para = dmrg_finite_size(para_dmrg)
#     save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
#             ('ob', 'a', 'info', 'para'))


with open('paras_ob.pkl', 'wb') as f_ob:
        pickle.dump(ob_list, f_ob)
with open('paras_info.pkl', 'wb') as f_info:
        pickle.dump(info_list, f_info)
np.savetxt('e0_per_site.txt', ob_e_persite_list, fmt='%.15g')