import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
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

parser.add_argument('--data_folder', default='/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/pTbin', help='Folder containing data and MC files')
parser.add_argument('--weights', default='/pscratch/sd/j/jing/H1PCT/weights', help='Folder to store trained weights')
parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')

parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('--plot_mom', action='store_true', default=False,help='Plot the moments')
parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')

parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--q2_int', type=int, default=0, help='Q2 interval to consider')
parser.add_argument('--img_fmt', default='png', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config = LoadJson(flags.config)

folder = 'results'
plot_folder = '/pscratch/sd/j/jing/H1PCT/scripts_perlmutter'
data_folder = flags.data_folder
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

colors = plt.cm.viridis(np.linspace(0, 1, 15))
npanels = 1

#xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
binning = np.linspace(0,1,50)
xaxis = 0.5*(binning[:-1] + binning[1:])
# print(xaxis)

pT_binning = opt.dedicated_binning['pt_c']
pT_x = 0.5*(pT_binning[:-1] + pT_binning[1:])
mc = 'Pythia'

if flags.plot_reco == True:
    gen_vars = ['tf_c','tf_f']
elif flags.plot_mom == True:
    gen_vars = ['central_mom']
else:
    gen_vars = ['gen_tf_c','gen_tf_f']

for var in gen_vars:

    if flags.plot_mom == True:
        data_folder = '/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/mom'
        cen_mom_q, cen_mom_g = {}, {}
        py_cen_mom_q, py_cen_mom_g = {}, {}
        # ax0.set_ylim(top=3)
        # ax1 = plt.subplot(gs[1],sharex=ax0)
        # ax1.set_ylim(top=1.1, bottom=0.9)
        key = 'cen_mom'

        # for each order
        for order in range(1,6):
            print(os.path.join(data_folder,"{}_{}.h5".format(var,order)))
            # input()
            f = h5.File(os.path.join(data_folder,"{}_{}.h5".format(var,order)),'r')
            cen_mom_q[order] = f[key+'_q'][:]
            cen_mom_g[order] = f[key+'_g'][:]
            py_cen_mom_q[order] = f['py_'+key+'_q'][:]
            py_cen_mom_g[order] = f['py_'+key+'_g'][:]

        # print("data quark:", cen_mom_q)
        # print("data gluon:", cen_mom_g)
        # print("mc quark:", py_cen_mom_q)                                            
        # print("mc gluon:", py_cen_mom_g)
        # input()
        for i in [4,5]:
            if i == 4:
                fig,gs = opt.SetGrid(npanels) 
                ax0 = plt.subplot(gs[0])
                ax0.plot(py_cen_mom_q[2]**2, py_cen_mom_q[4],color='r',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='qurark, k_4, Pythia')
                ax0.plot(py_cen_mom_g[2]**2, py_cen_mom_g[4],color='b',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_4, Pythia')

                ax0.plot(cen_mom_q[2]**2, cen_mom_q[4],color='g',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='qurark, k_4, Data')
                ax0.plot(cen_mom_g[2]**2, cen_mom_g[4],color='skyblue',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_4, Data')
                ax0.legend(loc='best',fontsize=10, ncol=2)
                ax0.set_xlim([-0.0005,0.004])
                ax0.set_ylim([-0.0039,0.0002])
                # for g
                # ax0.set_xlim([-0.0005,0.0008])
                # ax0.set_ylim([-0.0007,0.0001])
                fig.savefig(os.path.join(plot_folder,"central_mom_RG_flow_k4_k2sq.{}".format(flags.img_fmt)))

            else:
                fig,gs = opt.SetGrid(npanels) 
                ax0 = plt.subplot(gs[0])
                ax0.plot(py_cen_mom_q[3]*py_cen_mom_q[2], py_cen_mom_q[5],color='r',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='qurark, k_5, Pythia')
                ax0.plot(py_cen_mom_g[3]*py_cen_mom_g[2], py_cen_mom_g[5],color='b',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_5, Pythia')

                ax0.plot(cen_mom_q[3]*cen_mom_q[2], cen_mom_q[5],color='g',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='qurark, k_5, Data')
                ax0.plot(cen_mom_g[3]*cen_mom_g[2], cen_mom_g[5],color='skyblue',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_5, Data')
                ax0.legend(loc='best',fontsize=10, ncol=2)
                ax0.set_xlim([-0.00045,0.00008])
                ax0.set_ylim([-0.00025,0.0024])
                # for g
                # ax0.set_xlim([-0.00005,0.00002])
                # ax0.set_ylim([-0.00015,0.00022])
                fig.savefig(os.path.join(plot_folder,"central_mom_RG_flow_k5_k2k3.{}".format(flags.img_fmt)))

    # for full TF distribution
    else:
        hist = []
        fig,gs = opt.SetGrid(npanels) 
        ax0 = plt.subplot(gs[0])

        for q2_int in range(1,14):
            print(os.path.join(data_folder,"data_{}_{}.h5".format(var,q2_int)))
            # input()
            f = h5.File(os.path.join(data_folder,"data_{}_{}.h5".format(var,q2_int)),'r')
            hist.append(f[var][:])
            print(hist)
            # input()
            pt = pT_x[q2_int-1]
            ax0.plot(xaxis,hist[q2_int-1],color=colors[q2_int],marker='.',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=str(pt)+" GeV")
            ax0.legend(loc='best',fontsize=10, ncol=2)

        # ax1.plot(xaxis,np.divide(hist[-1], hist[0]),color='b',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='at pT 2250 /250')
        # ref = np.ones(1)
        # xmin,xmax = xaxis[0], xaxis[-1]
        # ax1.hlines(ref,xmin=xmin,xmax=xmax,linestyles='dashed',color='grey')#opt.sys_translate[sys])
        # ax1.legend()
        fig.savefig(os.path.join(plot_folder,"overlaid_data_{}_{}_{}.{}".format(var,flags.niter,flags.q2_int,flags.img_fmt)))