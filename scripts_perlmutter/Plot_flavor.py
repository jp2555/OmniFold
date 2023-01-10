import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import argparse
import os
import h5py as h5
from omnifold import  Multifold, Scaler, LoadJson
from SaveWeights import MCInfo
from scipy.stats import moment
import sys
import collections
sys.path.append('../')
import shared.options as opt


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/j/jing/h5', help='Folder containing data and MC files')
parser.add_argument('--weights', default='/pscratch/sd/j/jing/H1PCT/weights', help='Folder to store trained weights')
parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')


parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)

mc_names = ['Pythia_flavor']
# mc_names = ['Rapgap_nominal','Djangoh_nominal']
# standalone_predictions = ['Herwig','Sherpa']
standalone_predictions = []    
data_idx = 0 #Sample that after weights represent data
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1] #MC ref is used to define the reference simulation used to derive the closure and model systematic uncertainties
print(mc_ref)

version = data_name

if flags.plot_reco:
    gen_var_names = opt.reco_vars
else:
    gen_var_names = opt.gen_vars


def RatioLabel(ax1,var):
    ax1.set_ylabel('Rel. diff. [%]')
    ax1.set_xlabel(r'$p_T$ [GeV]')    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    # ax1.axhline(y=10, color='r', linestyle='--')
    # ax1.axhline(y=-10, color='r', linestyle='--')
    ylim = [-20,20]
    ax1.set_ylim(ylim)
    

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def compute_moment(values, weights, order):

    # first normalize the weights
    norm_wgt = weights / np.sum( weights, axis=0 )

    # raise event level value to a power, then add weights, then sum up (integrate dirac delta ftn)
    data_moment = np.sum( np.multiply( np.power(values, order), norm_wgt ), axis=0)

    return data_moment

def gluon_frac(values, weights):

    # first normalize the weights
    norm_wgt = weights / np.sum( weights, axis=0 )

    gluon = np.average( values, weights=weights) # value = 1 if gluon and 0 if not

    return gluon


def qg_splitting(f_cq, f_cg, f_fq, f_fg, Tc, Tf):

    a = ( f_cq*f_fg - f_fq*f_cg )**(-1) # coefficient of the inverse matrix
    Tq = a * (f_fg*Tc - f_cg*Tf)
    Tg = a * (-f_fq*Tc + f_cq*Tf)
    return (Tq, Tg)


def LoadData(q2_int):
    mc_info = {}
    # weights_data = {}
    # gen_q2 (pT) bin:  [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]
    # loading the bin:  gen_q2[q2_int-1] < pT < gen_q2[q2_int]
    # for q2_int in range(1,len(xaxis)+1)

    # mc_info['data'] = MCInfo('Data1516',flags.N,flags.data_folder,config,q2_int,is_reco=True)  
    #Loading weights from training
    for mc_name in mc_names:
        print("{}.h5".format(mc_name))    
        mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder,config,q2_int,is_reco=flags.plot_reco)
        if mc_name == data_name:
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)

            # weights_data[flags.mode] = mc_info[mc_name].ReturnWeights(flags.niter,model_name=model_name,mode=flags.mode)

    # weight_data = weights_data[flags.mode]
    return mc_info

################################################

# easy multi-key dict
id_c, id_f = [], []
diff = []

# for loading unfolded pT: [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]
binning = opt.dedicated_binning['pt_c']

# for plotting
xaxis = 0.5*(binning[:-1] + binning[1:])
print("binning", binning, "x axis", xaxis)
print("xaxis shape: ", len(xaxis))

# for faster debugging
# xaxis = [250, 350, 450, 550, 650, 750, 850, 950, 1100, 1300, 1500, 1800, 2250]
# input()

for var in gen_var_names:
    if not (var == 'gen_id_c' or var == 'gen_id_f' ): continue
    print(var)

    for q2_int in range(1,len(xaxis)+1):  # len of the pT bin list

        mc_info = LoadData(q2_int)
        pT = xaxis[q2_int-1] # pT bin center, corresponding to the q/g fraction dict
        print("pT: ", pT)

        # do only Pythia
        mc_name = mc_names[0]
        mc_var = mc_info[mc_name].LoadVar(var)
        # mask_var = np.abs(data_var)>=0

        if var == 'gen_id_c':
            g_frac_c = 100*gluon_frac(mc_var,weights=(mc_info[mc_name].nominal_wgts))
            print('g frac in central: ', g_frac_c)
            id_c.append( g_frac_c )
        elif var == 'gen_id_f':
            g_frac_f = 100*gluon_frac(mc_var,weights=(mc_info[mc_name].nominal_wgts))
            print('g frac in forward: ', g_frac_f)
            id_f.append( g_frac_f )

for i, f in enumerate(id_f):
    diff.append( id_c[i] - id_f[i] )

# plotting for each order
text_ypos = 0.85 #text position height
text_xpos = 0.22

fig,gs = opt.SetGrid(1) 
ax0 = plt.subplot(gs[0])
# ax0.tick_params(axis='x',labelsize=0)
# ax0.set_xscale('log')
opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'gluon fraction in %',ax0=ax0)
# ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
ax0.set_xlim(left=130,right=2.5e3)
ax0.plot(xaxis,id_c,color='r',label='central region', marker='+',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
ax0.plot(xaxis,id_f,color='b',label='forward region', marker='+',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
ax0.plot(xaxis,diff,color='k',label='difference (central-forward)', marker='+',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
ax0.legend(loc='upper left',fontsize=16,ncol=2)    

plot_folder = '../plots_gluon_frac'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
fig.savefig(os.path.join(plot_folder,"gluon_frac_{}.{}".format(flags.niter,flags.img_fmt)))
