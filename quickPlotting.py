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
import sys
import collections # to have nested dicts
sys.path.append('../')
import shared.options as opt


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/j/jing/H1PCT/scripts_perlmutter', help='Folder containing data and MC files')
parser.add_argument('--weights', default='/pscratch/sd/j/jing/H1PCT/weights', help='Folder to store trained weights')
parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')

parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('--plot_ratio', action='store_true', default=False,help='Plot the pT_trk/pT_all ratio instead of pT alone')
parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')

parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--q2_int', type=int, default=0, help='Q2 interval to consider')
parser.add_argument('--img_fmt', default='png', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config = LoadJson(flags.config)

folder = 'results'
plot_folder = '/pscratch/sd/j/jing/H1PCT/quickPlotting'
text_ypos = 0.67 #text position height
text_xpos = 0.82

def RatioLabel(ax1,var):
    ax1.set_ylabel('Rel. diff. [%]')
    ax1.set_xlabel(gen_var_names[var])    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    # ax1.axhline(y=10, color='r', linestyle='--')
    # ax1.axhline(y=-10, color='r', linestyle='--')
    ylim = [-70,70]
    ax1.set_ylim(ylim)

fig,gs = opt.SetGrid(npanels) 
ax0 = plt.subplot(gs[0])
ax0.set_ylim(top=3)
ax1 = plt.subplot(gs[1],sharex=ax0)
#xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
binning = opt.dedicated_binning['gen_pt_c']
xaxis = 0.5*(binning[:-1] + binning[1:])
mc = 'Pythia'

gen_vars = ['gen_pt_c','gen_pt_f']
    
for q2_int in range(0,14):
    for var in gen_vars:
        hist = h5.File(os.path.join(data_folder,"{}_{}.h5".format(var,q2_int)),'r')[1]

        ax0.plot(xaxis,hist,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
        fig.savefig(os.path.join(plot_folder,"{}_{}_{}.{}".format(var,flags.niter,flags.q2_int,flags.img_fmt)))