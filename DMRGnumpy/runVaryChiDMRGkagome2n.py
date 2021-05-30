import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')      # ~/SynologyDrive/QuCNN/T-Nalg
from DMRGalgorithms import dmrg_finite_size
import Parameters as Pm
import Qubism
import numpy as np
from os import path
from BasicFunctions import load_pr, save_pr, plot, output_txt, print_dict, mkdir
from TensorBasicModule import open_mps_product_state_spin_up
import MPSClass
import pickle
import os

paras = np.arange(0.4, 1.0, 0.0006)

lattice = 'kagome2n'
para_dmrg = Pm.generate_parameters_dmrg(lattice)
para_dmrg['spin'] = 'half'
para_dmrg['num_columns'] = 2
para_dmrg['bound_cond'] = 'pbc'
para_dmrg['chi'] = 128
para_dmrg['jxy'] = 1
para_dmrg['jz'] = 1
para_dmrg['hx'] = 0
para_dmrg['hz'] = 0
para_dmrg['project_path'] = '.'
#para_dmrg['data_path'] = 'data/HeisenbergKagome2n'

para_dmrg['data_path'] = os.path.join('dataQubism', 'states')
para_dmrg['image_path'] = os.path.join('dataQubism', 'images')

mkdir(para_dmrg['data_path'])
mkdir(para_dmrg['image_path'])

info_list, ob_list = [], []
ob_e_persite_list = []
for i in range(len(paras)):
    para_dmrg['ratio'] = paras[i]
    para_dmrg = Pm.make_consistent_parameter_dmrg_2(para_dmrg)
    #print_dict(para_dmrg)
    #ob, a, info, para = dmrg_finite_size(para_dmrg)
    #print('Energy per site = ' + str(ob['e_per_site']))
    #save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para), ('ob', 'a', 'info', 'para'))
    ob, a, info, para = dmrg_finite_size(para_dmrg)
    save_pr(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr', (ob, a, info, para), ('ob', 'a', 'info', 'para'))
    
    ob_list.append(ob)
    info_list.append(info)
    ob_e_persite_list.append(ob['e_per_site'])
    
    state = a.full_coefficients_mps()
    image = Qubism.state2image(state * 256, para['d'], is_rescale=True)
    # image = Image.fromarray(image.astype(np.uint8))
    image = Qubism.image2rgb(image, if_rescale_1=False)
    image.save(os.path.join(para_dmrg['image_path'], para_dmrg['data_exp'] + '.jpg'))


with open('paras_ob.pkl', 'wb') as f_ob:
        pickle.dump(ob_list, f_ob)
with open('paras_info.pkl', 'wb') as f_info:
        pickle.dump(info_list, f_info)
np.savetxt('e0_per_site.txt', ob_e_persite_list, fmt='%.15g')