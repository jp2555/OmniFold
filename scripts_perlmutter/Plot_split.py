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

parser.add_argument('--data_folder', default='/pscratch/sd/j/jing/H1PCT/scripts_perlmutter', help='Folder containing data and MC files')
parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')

parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('--ibu', action='store_true', default=False,help='Plot IBU results')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('--plot_flow', action='store_true', default=False,help='Plot the RGflow mixing')
parser.add_argument('--plot_mom', action='store_true', default=False,help='Plot the moments')
parser.add_argument('--plot_dist', action='store_true', default=False,help='Plot the overlaid distributions')
parser.add_argument('--plot_n', action='store_true', default=False,help='Plot the event / bin count')
parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')

parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--q2_int', type=int, default=0, help='Q2 interval to consider')
parser.add_argument('--ptmax', type=int, default=0, help='Max pT to consider')
parser.add_argument('--img_fmt', default='png', help='Format of the output figures')
parser.add_argument('--upTo', type=int, default=6, help='Highest order of the moment to compute to')

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

def compute_moment(values, weights, order):

    # first normalize the weights
    norm_wgt = weights / np.sum( weights, axis=0 )

    # raise event level value to a power, then add weights, then sum up (integrate dirac delta ftn)
    data_moment = np.sum( np.multiply( np.power(values, order), norm_wgt ), axis=0)

    return data_moment


def qg_splitting(f_cq, f_cg, f_fq, f_fg, Tc, Tf):

    a = ( f_cq*f_fg - f_fq*f_cg )**(-1) # coefficient of the inverse matrix
    Tq = a * (f_fg*Tc - f_cg*Tf)
    Tg = a * (-f_fq*Tc + f_cq*Tf)
    return (Tq, Tg)

binning = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500])
pT_x = 0.5*(binning[:-1] + binning[1:])
mc = 'Pythia'

################################################
# Processing cumulants for RG flow mixing plots
################################################

mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

sys_mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
sys_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

py_mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
py_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
py_cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

coefs = {
    # GLUON fraction i.e (f_cg, f_fg)
    # central (large), forward (small)
    # from fragmentation measurement 
    250: (66.2, 56.7), 350: (62.1, 50.8),
    450: (59.3, 46.3), 550: (56.2, 42.3),
    650: (52.9, 39.4), 750: (50.5, 36.9),
    850: (47.6, 34.7), 950: (45.2, 32.8),
    1100: (42.01, 30.50), 1300: (37.55, 27.48), 
    1500: (34.10, 25.18), 1800: (30.07, 21.73), 
    2250: (23.31, 17.41),
    }
    # Fron Ntuples
    # 250: (64.86, 60.11), 350: (61.14, 55.39),
    # 450: (57.62, 51.12), 550: (54.36, 47.66),
    # 650: (51.44, 44.74), 750: (48.71, 42.25),
    # 850: (46.05, 40.02), 950: (43.70, 38.11),
    # 1100: (40.71, 35.88), 1300: (36.64, 32.72), 
    # 1500: (32.92, 30.07), 1800: (28.85, 27.11), 
    # 2250: (22.95, 21.99),
    # }

gen_vars = ['gen_tf_c','gen_tf_f']
flavors = ['gen_tf_q','gen_tf_g']
orders = range(1,flags.upTo+1)
# variations = ['nominal','sys']


if flags.ibu:

    for order in orders: # one list / order
        # Reading IBU results
        for i,flavor in enumerate(['gen_tf_q','gen_tf_g']): # one list / order

            print(os.path.join('/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/mom_sys',"{}_{}.h5".format(flavor,order)))
            f = h5.File(os.path.join('/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/mom_sys',"{}_{}.h5".format(flavor,order)),'r')
            pT_x = np.array(f['xaxis'][:])
            if i == 0:
                q_data_ibu = np.array(f['nominal'][:])
                q_fine_data_ibu = np.array(f['m_fine'][:])
                q_py_ibu = np.array(f['coarse'][:])
                q_fine_py_ibu = np.array(f['fine'][:])
                q_corr_ibu = np.array(f['corr'][:])
                if flags.sys:
                    q_sys_ibu = np.array(f['sys_'+flavor][:])
            else:
                g_data_ibu = np.array(f['nominal'][:])
                g_fine_data_ibu = np.array(f['m_fine'][:])
                g_py_ibu = np.array(f['coarse'][:])
                g_fine_py_ibu = np.array(f['fine'][:])
                g_corr_ibu = np.array(f['corr'][:])
                if flags.sys:
                    g_sys_ibu = np.array(f['sys_'+flavor][:])

        fig,gs = opt.SetGrid(1) 
        ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1],sharex=ax0)
        # ax0.set_xscale('log')
        opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'order+$th moment$',ax0=ax0)
        ax0.set_xlim(left=130,right=2.5e3)
        text_xpos = 0.78 #text position height
        text_ypos = 0.7     
            
        plt.text(text_xpos, text_ypos,
                 '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax0.transAxes, fontsize=18)

        #ax0.set_ylim(top=1.6,bottom = 0.)
        ax0.set_ylim(top=2*max(q_data_ibu),bottom = 0.)

        # print("x axis: ", pT_x, "data size: ", len(data_ibu), "err: ", np.abs(data_ibu)*sys_unc/100.0 )
        # sqrt of sys already taken when writing into the sys_dict above #L262
        # ax0.plot(pT_x, q_data_ibu, yerr = np.abs(q_data_ibu*q_sys_unc/100), fmt='o', ms=12, color='b', label='Data')
        
        #ax0.step(pT_x, q_py_ibu, where='mid',color='skyblue',label='coarse, Quark Pythia 8.230 A14')
        ax0.plot(pT_x, q_fine_py_ibu,color='skyblue',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Quark Pythia 8.230 A14')
        ax0.plot(pT_x, g_fine_py_ibu,color='pink',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Gluon Pythia 8.230 A14')
        #ax0.plot(pT_x, q_corr_ibu, marker='o',mfc='b', ms=12, color='b', label='corrected IBU Quark Data',linestyle="None") 
        plt.step(pT_x, q_data_ibu, where='mid',color='b',linestyle='--',label='IBU Quark Data')
        plt.step(pT_x, g_data_ibu,where='mid',color='r',linestyle='--',label='IBU Gluon Data')
        #plt.step(pT_x, q_fine_data_ibu, where='mid',color='navy',linestyle='--',label='fine IBU Quark Data')
        #plt.step(pT_x, q_fine_py_ibu,where='mid',color='navy',label='Quark Pythia 8.230 A14')
        #ax0.plot(pT_x, q_fine_py_ibu,color='skyblue',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Quark Pythia 8.230 A14')

        #ax0.plot(pT_x, g_fine_py_ibu,color='pink',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Gluon Pythia 8.230 A14')
        
        ax0.plot(pT_x, q_corr_ibu, marker='o',mfc='b', ms=12, color='b', label='corrected IBU Quark Data',linestyle="None")
        #plt.step(pT_x, g_fine_py_ibu,where='mid',color='brown',label='Gluon Pythia 8.230 A14')
        #ax0.step(pT_x, g_py_ibu,where='mid',color='pink',label='coarse, Gluon Pythia 8.230 A14')
        #plt.step(pT_x, g_fine_data_ibu, where='mid',color='brown',linestyle='--',label='fine IBU Gluon Data')
        ax0.plot(pT_x, g_corr_ibu, marker='o', mfc='r',ms=12, color='r', label='corrected IBU Gluon Data',linestyle="None") 
        #plt.step(pT_x, g_data_ibu,where='mid',color='r',linestyle='--',label='IBU Gluon Data')

        ax0.legend(loc='upper left',fontsize=16) 

        if flags.sys:
            for ibin in range(len(pT_x)):
                xup = binning[ibin+1]
                xlow = binning[ibin] 
                # ax0.hlines(y=q_py_ibu[ibin], xmin=xlow, xmax=xup, colors='b', label='Quark Pythia 8.230 A14')
                # ax0.hlines(y=g_py_ibu[ibin], xmin=xlow, xmax=xup, colors='r', label='Gluon Pythia 8.230 A14')
                ax0.fill_between(np.array([xlow,xup]),q_data_ibu[ibin]+np.abs(q_data_ibu[ibin]*q_sys_unc[ibin]/100),q_data_ibu[ibin]-np.abs(q_data_ibu[ibin]*q_sys_unc[ibin]/100), alpha=0.2,color='skyblue')
                ax0.fill_between(np.array([xlow,xup]),g_data_ibu[ibin]+np.abs(g_data_ibu[ibin]*g_sys_unc[ibin]/100),g_data_ibu[ibin]-np.abs(g_data_ibu[ibin]*g_sys_unc[ibin]/100), alpha=0.2,color='pink')

        plot_folder = '../scripts_perlmutter/plots_mom_ibu'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"{}_{}.{}".format(flags.niter,order,flags.img_fmt)))


# Load saved data
for var in gen_vars:

    for order in orders:
        print(os.path.join(data_folder,"{}_{}.h5".format(var,order)))
        # input()
        f = h5.File(os.path.join(data_folder,"{}_{}.h5".format(var,order)),'r')
        pT_xaxis = np.array(f['xaxis'][:])
        data_pred = np.array(f[var][:])
        py_pred = np.array(f['py_'+var][:])
        if flags.sys:
            sys_pred = np.array(f['sys_'+var][:])
        
        # print(data_pred, py_pred, sys_pred)
        for i, pTbin in enumerate(pT_xaxis):
            mom_cf[pTbin][var]['nominal'][order] = data_pred[i]
            py_mom_cf[pTbin][var]['nominal'][order] = py_pred[i]
            if flags.sys:
                sys_mom_cf[pTbin][var]['sys'][order] = sys_pred[i]

    # print(mom_cf, py_mom_cf, sys_mom_cf)

# Split q/g
for order in orders:
    for q2_int in range(1,len(pT_x)+1):
        pT_coef = pT_x[q2_int-1] # here fore reading q/g fraction, same as the pT key
        print(pT_coef, ' GeV')

        f_cg, f_fg = coefs[pT_coef][0]*0.01, coefs[pT_coef][1]*0.01
        f_cq, f_fq = 1-f_cg, 1-f_fg
        # print("f_cg, f_fg, f_cq, f_fq: ", f_cg, f_fg, f_cq, f_fq)

        # print( "Tc", mom_cf[pT_coef]['gen_tf_c'][variation])
        Tc = mom_cf[pT_coef]['gen_tf_c']['nominal'][order]
        Tf = mom_cf[pT_coef]['gen_tf_f']['nominal'][order]
        Tqg = qg_splitting(f_cq, f_cg, f_fq, f_fg, Tc, Tf) 
        # print(Tqg)
       
        mom_qg['nominal'][order]['gen_tf_q'][pT_coef] = Tqg[0] #np.abs(Tqg[0])
        mom_qg['nominal'][order]['gen_tf_g'][pT_coef] = Tqg[1] #np.abs(Tqg[1])

        py_Tc = py_mom_cf[pT_coef]['gen_tf_c']['nominal'][order]
        py_Tf = py_mom_cf[pT_coef]['gen_tf_f']['nominal'][order]
        py_Tqg = qg_splitting(f_cq, f_cg, f_fq, f_fg, py_Tc, py_Tf) 

        py_mom_qg['nominal'][order]['gen_tf_q'][pT_coef] = py_Tqg[0] #np.abs(py_Tqg[0])
        py_mom_qg['nominal'][order]['gen_tf_g'][pT_coef] = py_Tqg[1] #np.abs(py_Tqg[1])

        if flags.sys:
            sys_Tc = sys_mom_cf[pT_coef]['gen_tf_c']['sys'][order]
            sys_Tf = sys_mom_cf[pT_coef]['gen_tf_f']['sys'][order]
            sys_Tqg = qg_splitting(f_cq, f_cg, f_fq, f_fg, sys_Tc, sys_Tf) 

            sys_mom_qg['sys'][order]['gen_tf_q'][pT_coef] = sys_Tqg[0] #np.abs(sys_Tqg[0])
            sys_mom_qg['sys'][order]['gen_tf_g'][pT_coef] = sys_Tqg[1] #np.abs(sys_Tqg[1])

print("moment q/g:", mom_qg)
# print("sys moment q/g:", sys_mom_qg)
print("Pythia moment q/g:", py_mom_qg)
# input()

for flavor in flavors:

    for order in orders: # one list / order
        fig,gs = opt.SetGrid(2) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1],sharex=ax0)
        # ax0.set_xscale('log')
        opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'order+$th moment$',ax0=ax0)
        # ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
        ax0.set_xlim(left=130,right=2.5e3)
        # ax0.set_ylim(bottom=0.,top=1.)
        # ax0.legend(loc='upper left',fontsize=16,ncol=2)  
        text_ypos = 0.85 #text position height
        text_xpos = 0.22      
            
        plt.text(text_xpos, text_ypos,
                 '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax0.transAxes, fontsize=18)

        data_pred, py_pred = [], []
        sys_unc = []
        mc = 'Pythia'

        for q2_int in range(1,len(pT_x)+1):
            pT = pT_x[q2_int-1]
            data_pred.append( mom_qg['nominal'][order][flavor][pT] )
            py_pred.append( py_mom_qg['nominal'][order][flavor][pT] )
            if flags.sys:
                sys_unc.append( sys_mom_qg['sys'][order][flavor][pT] )

        data_pred = np.array( data_pred )
        sys_unc = np.array( np.abs(sys_unc) )
        py_pred = np.array( py_pred )
        # print('To plot: ', data_pred, py_pred, sys_unc)
        ax0.set_ylim(top=2*max(data_pred),bottom = 0.)

        if flags.sys == True:
            print("x axis: ", pT_x, "data size: ", len(data_pred), "err: ", np.abs(data_pred)*sys_unc/100.0 )
            # sqrt of sys already taken when writing into the sys_dict above #L262
            ax0.errorbar(pT_x, data_pred, yerr = np.abs(data_pred*sys_unc/100), fmt='o', ms=12, color='k', label='Data') 
            ax0.plot(pT_x,py_pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
        else:
            ax0.plot(pT_x,py_pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
            ax0.plot(pT_x,data_pred, marker='o', lw=0, ms=12, color='k',label='Data')

        if flags.sys:
            for ibin in range(len(pT_x)):
                xup = binning[ibin+1]
                xlow = binning[ibin] 
                ax0.hlines(y=data_pred[ibin], xmin=xlow, xmax=xup, colors='black')
                ax1.fill_between(np.array([xlow,xup]),np.sqrt(sys_unc[ibin]),-np.sqrt(sys_unc[ibin]), alpha=0.3,color='k')

        # for mc_name in mc_names:
        #     mc = mc_name.split("_")[0]
        # ax0.plot(pT_x,py_pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
        ratio = np.divide(py_pred-data_pred,data_pred)
        ax1.plot(pT_x,ratio,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

        plot_folder = '../scripts_perlmutter/plots_mom_'+flavor
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"{}_{}_{}.{}".format(flavor,flags.niter,order,flags.img_fmt)))


######################

# Start optional plotting

######################

if flags.plot_mom:

    for order in orders: # one list / order
        fig,gs = opt.SetGrid(1) 
        ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1],sharex=ax0)
        # ax0.set_xscale('log')
        opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'order+$th moment$',ax0=ax0)
        # ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
        ax0.set_xlim(left=130,right=2.5e3)
        # ax0.set_ylim(bottom=0.,top=1.)
        # ax0.legend(loc='upper left',fontsize=16,ncol=2)  
        text_ypos = 0.7 #text position height
        text_xpos = 0.78      
            
        plt.text(text_xpos, text_ypos,
                 '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax0.transAxes, fontsize=18)

        q_data_pred, q_py_pred = [], []
        g_data_pred, g_py_pred = [], []
        if flags.sys:
            q_sys_unc = []
            g_sys_unc = []
        mc = 'Pythia'

        for q2_int in range(1,len(pT_x)+1):
            pT = pT_x[q2_int-1]
            q_data_pred.append( mom_qg['nominal'][order]['gen_tf_q'][pT] )
            q_py_pred.append( py_mom_qg['nominal'][order]['gen_tf_q'][pT] )

            g_data_pred.append( mom_qg['nominal'][order]['gen_tf_g'][pT] )
            g_py_pred.append( py_mom_qg['nominal'][order]['gen_tf_g'][pT] )

            if flags.sys:
                q_sys_unc.append( sys_mom_qg['sys'][order]['gen_tf_q'][pT] )
                g_sys_unc.append( sys_mom_qg['sys'][order]['gen_tf_g'][pT] )

        q_data_pred = np.array( q_data_pred )
        q_py_pred = np.array( q_py_pred )

        g_data_pred = np.array( g_data_pred )
        g_py_pred = np.array( g_py_pred )

        if flags.sys:
            q_sys_unc = np.array( np.abs(q_sys_unc) )
            g_sys_unc = np.array( np.abs(g_sys_unc) )
        # print('To plot: ', data_pred, py_pred, sys_unc)
        ax0.set_ylim(top=2*max(q_data_pred),bottom = 0.)

        # print("x axis: ", pT_x, "data size: ", len(data_pred), "err: ", np.abs(data_pred)*sys_unc/100.0 )
        # sqrt of sys already taken when writing into the sys_dict above #L262
        # ax0.plot(pT_x, q_data_pred, yerr = np.abs(q_data_pred*q_sys_unc/100), fmt='o', ms=12, color='b', label='Data')
        ax0.plot(pT_x,q_py_pred,color='skyblue',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Quark Pythia 8.230 A14')
        ax0.plot(pT_x,g_py_pred,color='pink',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Gluon Pythia 8.230 A14')

        ax0.plot(pT_x, q_data_pred, marker='o',mfc='b', ms=12, color='b', label='Quark Data',linestyle="None") 
        
        #ax0.plot(pT_x,g_py_pred,color='pink',marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='Gluon Pythia 8.230 A14')
        ax0.plot(pT_x, g_data_pred, marker='o', mfc='r',ms=12, color='r', label='Gluon Data',linestyle="None") 

        ax0.legend(loc='upper left',fontsize=16)#,ncol=2) 

        if flags.sys:
            for ibin in range(len(pT_x)):
                xup = binning[ibin+1]
                xlow = binning[ibin] 
                # ax0.hlines(y=q_py_pred[ibin], xmin=xlow, xmax=xup, colors='b', label='Quark Pythia 8.230 A14')
                # ax0.hlines(y=g_py_pred[ibin], xmin=xlow, xmax=xup, colors='r', label='Gluon Pythia 8.230 A14')
                ax0.fill_between(np.array([xlow,xup]),q_data_pred[ibin]+np.abs(q_data_pred[ibin]*q_sys_unc[ibin]/100),q_data_pred[ibin]-np.abs(q_data_pred[ibin]*q_sys_unc[ibin]/100), alpha=0.2,color='skyblue')
                ax0.fill_between(np.array([xlow,xup]),g_data_pred[ibin]+np.abs(g_data_pred[ibin]*g_sys_unc[ibin]/100),g_data_pred[ibin]-np.abs(g_data_pred[ibin]*g_sys_unc[ibin]/100), alpha=0.2,color='pink')

        # for mc_name in mc_names:
        #     mc = mc_name.split("_")[0]
        # ax0.plot(pT_x,py_pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
        # ratio = np.divide(py_pred-data_pred,data_pred)
        # ax1.plot(pT_x,ratio,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

        plot_folder = '../scripts_perlmutter/plots_mom_both'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"{}_{}.{}".format(flags.niter,order,flags.img_fmt)))


######################

if flags.plot_flow:
    for order in orders:
        cen_mom_q, cen_mom_g = [], []
        py_cen_mom_q, py_cen_mom_g = [], []
        # sys_cen_mom_q, sys_cen_mom_g = [], []

        for q2_int in range(1,len(pT_x)+1):
            pT_coef = pT_x[q2_int-1]
            Tq1, Tg1 = mom_qg['nominal'][1]['gen_tf_q'][pT_coef], mom_qg['nominal'][1]['gen_tf_g'][pT_coef]
            Tq2, Tg2 = mom_qg['nominal'][2]['gen_tf_q'][pT_coef], mom_qg['nominal'][2]['gen_tf_g'][pT_coef]
            Tq3, Tg3 = mom_qg['nominal'][3]['gen_tf_q'][pT_coef], mom_qg['nominal'][3]['gen_tf_g'][pT_coef]
            Tq4, Tg4 = mom_qg['nominal'][4]['gen_tf_q'][pT_coef], mom_qg['nominal'][4]['gen_tf_g'][pT_coef]
            Tq5, Tg5 = mom_qg['nominal'][5]['gen_tf_q'][pT_coef], mom_qg['nominal'][5]['gen_tf_g'][pT_coef]
            Tq6, Tg6 = mom_qg['nominal'][6]['gen_tf_q'][pT_coef], mom_qg['nominal'][6]['gen_tf_g'][pT_coef]

            if order == 1:
                cen_mom_q.append( Tq1)
                cen_mom_g.append( Tg1)
            if order == 2:
                cen_mom_q.append( Tq2 - Tq1**2)
                cen_mom_g.append( Tg2 - Tg1**2)
            if order == 3:
                cen_mom_q.append( Tq3 - 3*Tq2*Tq1 + 2*Tq1**3)
                cen_mom_g.append( Tg3 - 3*Tg2*Tg1 + 2*Tg1**3)
            if order == 4:
                cen_mom_q.append( Tq4 - 4*Tq3*Tq1 - 3*Tq2**2 + 12*Tq2*Tq1**2 -6*Tq1**4)
                cen_mom_g.append( Tg4 - 4*Tg3*Tg1 - 3*Tg2**2 + 12*Tg2*Tg1**2 -6*Tg1**4)
            if order == 5:
                cen_mom_q.append( Tq5 - 5*Tq4*Tq1 - 10*Tq3*Tq2 + 20*Tq3*Tq1**2 + 30*(Tq2**2)*Tq1 - 60*Tq2*Tq1**3 + 24*Tq1**5)
                cen_mom_g.append( Tg5 - 5*Tg4*Tg1 - 10*Tg3*Tg2 + 20*Tg3*Tg1**2 + 30*(Tg2**2)*Tg1 - 60*Tg2*Tg1**3 + 24*Tg1**5)
            if order == 6:
                cen_mom_q.append( Tq6-6*Tq5*Tq1-15*Tq4*Tq2+30*Tq4*(Tq1**2)-10*(Tq3**2)+120*Tq3*Tq2*Tq1-120*Tq3*(Tq1**3)+30*(Tq2**3)-270*(Tq2**2)*(Tq1**2)+360*Tq2*(Tq1**4)-120*(Tq1**6) )
                cen_mom_g.append( Tg6-6*Tg5*Tg1-15*Tg4*Tg2+30*Tg4*(Tg1**2)-10*(Tg3**2)+120*Tg3*Tg2*Tg1-120*Tg3*(Tg1**3)+30*(Tg2**3)-270*(Tg2**2)*(Tg1**2)+360*Tg2*(Tg1**4)-120*(Tg1**6))

            # for Pythia
            py_Tq1, py_Tg1 = py_mom_qg['nominal'][1]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][1]['gen_tf_g'][pT_coef]
            py_Tq2, py_Tg2 = py_mom_qg['nominal'][2]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][2]['gen_tf_g'][pT_coef]
            py_Tq3, py_Tg3 = py_mom_qg['nominal'][3]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][3]['gen_tf_g'][pT_coef]
            py_Tq4, py_Tg4 = py_mom_qg['nominal'][4]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][4]['gen_tf_g'][pT_coef]
            py_Tq5, py_Tg5 = py_mom_qg['nominal'][5]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][5]['gen_tf_g'][pT_coef]
            py_Tq6, py_Tg6 = py_mom_qg['nominal'][6]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][6]['gen_tf_g'][pT_coef]

            if order == 1:
                py_cen_mom_q.append( py_Tq1)
                py_cen_mom_g.append( py_Tg1)
            if order == 2:
                py_cen_mom_q.append( py_Tq2 - py_Tq1**2)
                py_cen_mom_g.append( py_Tg2 - py_Tg1**2)
            if order == 3:
                py_cen_mom_q.append( py_Tq3 - 3*py_Tq2*py_Tq1 + 2*py_Tq1**3)
                py_cen_mom_g.append( py_Tg3 - 3*py_Tg2*py_Tg1 + 2*py_Tg1**3)
            if order == 4:
                py_cen_mom_q.append( py_Tq4 - 4*py_Tq3*py_Tq1 - 3*py_Tq2**2 + 12*py_Tq2*py_Tq1**2 -6*py_Tq1**4)
                py_cen_mom_g.append( py_Tg4 - 4*py_Tg3*py_Tg1 - 3*py_Tg2**2 + 12*py_Tg2*py_Tg1**2 -6*py_Tg1**4)
            if order == 5:
                py_cen_mom_q.append( py_Tq5 - 5*py_Tq4*py_Tq1 - 10*py_Tq3*py_Tq2 + 20*py_Tq3*py_Tq1**2 + 30*(py_Tq2**2)*py_Tq1 - 60*py_Tq2*py_Tq1**3 + 24*py_Tq1**5)
                py_cen_mom_g.append( py_Tg5 - 5*py_Tg4*py_Tg1 - 10*py_Tg3*py_Tg2 + 20*py_Tg3*py_Tg1**2 + 30*(py_Tg2**2)*py_Tg1 - 60*py_Tg2*py_Tg1**3 + 24*py_Tg1**5)
            if order == 6:
                py_cen_mom_q.append( py_Tq6-6*py_Tq5*py_Tq1-15*py_Tq4*py_Tq2+30*py_Tq4*(py_Tq1**2)-10*(py_Tq3**2)+120*py_Tq3*py_Tq2*py_Tq1-120*py_Tq3*(py_Tq1**3)+30*(py_Tq2**3)-270*(py_Tq2**2)*(py_Tq1**2)+360*py_Tq2*(py_Tq1**4)-120*(py_Tq1**6) )
                py_cen_mom_g.append( py_Tg6-6*py_Tg5*py_Tg1-15*py_Tg4*py_Tg2+30*py_Tg4*(py_Tg1**2)-10*(py_Tg3**2)+120*py_Tg3*py_Tg2*py_Tg1-120*py_Tg3*(py_Tg1**3)+30*(py_Tg2**3)-270*(py_Tg2**2)*(py_Tg1**2)+360*py_Tg2*(py_Tg1**4)-120*(py_Tg1**6))


            # for sys
            # sys_Tq1, sys_Tg1 = sys_mom_qg['sys'][1]['gen_tf_q'][pT_coef], sys_mom_qg['sys'][1]['gen_tf_g'][pT_coef]
            # sys_Tq2, sys_Tg2 = sys_mom_qg['sys'][2]['gen_tf_q'][pT_coef], sys_mom_qg['sys'][2]['gen_tf_g'][pT_coef]
            # sys_Tq3, sys_Tg3 = sys_mom_qg['sys'][3]['gen_tf_q'][pT_coef], sys_mom_qg['sys'][3]['gen_tf_g'][pT_coef]
            # sys_Tq4, sys_Tg4 = sys_mom_qg['sys'][4]['gen_tf_q'][pT_coef], sys_mom_qg['sys'][4]['gen_tf_g'][pT_coef]
            # sys_Tq5, sys_Tg5 = sys_mom_qg['sys'][5]['gen_tf_q'][pT_coef], sys_mom_qg['sys'][5]['gen_tf_g'][pT_coef]

            # if order == 1:
            #     sys_cen_mom_q.append( sys_Tq1)
            #     sys_cen_mom_g.append( sys_Tg1)
            # if order == 2:
            #     sys_cen_mom_q.append( sys_Tq2 - sys_Tq1**2)
            #     sys_cen_mom_g.append( sys_Tg2 - sys_Tg1**2)
            # if order == 3:
            #     sys_cen_mom_q.append( sys_Tq3 - 3*sys_Tq2*sys_Tq1 + 2*sys_Tq1**3)
            #     sys_cen_mom_g.append( sys_Tg3 - 3*sys_Tg2*sys_Tg1 + 2*sys_Tg1**3)
            # if order == 4:
            #     sys_cen_mom_q.append( sys_Tq4 - 4*sys_Tq3*sys_Tq1 - 3*sys_Tq2**2 + 12*sys_Tq2*sys_Tq1**2 -6*sys_Tq1**4)
            #     sys_cen_mom_g.append( sys_Tg4 - 4*sys_Tg3*sys_Tg1 - 3*sys_Tg2**2 + 12*sys_Tg2*sys_Tg1**2 -6*sys_Tg1**4)
            # if order == 5:
            #     sys_cen_mom_q.append( sys_Tq5 - 5*sys_Tq4*sys_Tq1 - 10*sys_Tq3*sys_Tq2 + 20*sys_Tq3*sys_Tq1**2 + 30*(sys_Tq2**2)*sys_Tq1 - 60*sys_Tq2*sys_Tq1**3 + 24*sys_Tq1**5)
            #     sys_cen_mom_g.append( sys_Tg5 - 5*sys_Tg4*sys_Tg1 - 10*sys_Tg3*sys_Tg2 + 20*sys_Tg3*sys_Tg1**2 + 30*(sys_Tg2**2)*sys_Tg1 - 60*sys_Tg2*sys_Tg1**3 + 24*sys_Tg1**5)

        cen_mom_q, cen_mom_g = np.array( cen_mom_q ), np.array( cen_mom_g )
        py_cen_mom_q, py_cen_mom_g = np.array( py_cen_mom_q ), np.array( py_cen_mom_g )
        # sys_cen_mom_q, sys_cen_mom_g = np.array( sys_cen_mom_q ), np.array( sys_cen_mom_g )

        text_ypos = 0.85 #text position height
        text_xpos = 0.22

        toSave = np.zeros( (len(binning)-1, 5))
        toSave[:, 0] = pT_x
        toSave[:, 1] = cen_mom_q
        toSave[:, 2] = cen_mom_g
        toSave[:, 3] = py_cen_mom_q
        toSave[:, 4] = py_cen_mom_g
        f = h5.File("central_mom_"+str(order)+'.h5', 'w')
        f['xaxis']=toSave[:,0]
        f['cen_mom_q']=toSave[:, 1]
        f['cen_mom_g']=toSave[:, 2]
        f['py_cen_mom_q']=toSave[:, 3]
        f['py_cen_mom_g']=toSave[:, 4]
        f.close()  

        # fig,gs = opt.SetGrid(1) 
        # ax0 = plt.subplot(gs[0])
        # # ax0.tick_params(axis='x',labelsize=0)
        # # ax0.set_xscale('log')
        # opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$p_\mathrm{T}^\mathrm{charged}/p_\mathrm{T}^\mathrm{all}$ central moments',ax0=ax0)
        # # ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
        # ax0.set_xlim(left=130,right=1.5e3)
        # ax0.plot(pT_x[0:flags.ptmax-1],cen_mom_q,color='b',label='data_qurak')
        # ax0.plot(pT_x[0:flags.ptmax-1],cen_mom_g,color='r',label='data_gluon')
        # ax0.plot(pT_x[0:flags.ptmax-1],py_cen_mom_q,color='skyblue',label='Pythia_qurak')
        # ax0.plot(pT_x[0:flags.ptmax-1],py_cen_mom_g,color='coral',label='Pythia_gluon')
        # for ibin in range(flags.ptmax-1):
        #         xup = binning[ibin+1]
        #         xlow = binning[ibin] 
        #         # ax0.hlines(y=q_py_pred[ibin], xmin=xlow, xmax=xup, colors='b', label='Quark Pythia 8.230 A14')
        #         # ax0.hlines(y=g_py_pred[ibin], xmin=xlow, xmax=xup, colors='r', label='Gluon Pythia 8.230 A14')
        #         ax0.fill_between(np.array([xlow,xup]),cen_mom_q[ibin]+np.abs(cen_mom_q[ibin]*sys_cen_mom_q[ibin]/100),cen_mom_q[ibin]-np.abs(cen_mom_q[ibin]*sys_cen_mom_q[ibin]/100), alpha=0.2,color='skyblue')
        #         ax0.fill_between(np.array([xlow,xup]),cen_mom_g[ibin]+np.abs(cen_mom_g[ibin]*sys_cen_mom_g[ibin]/100),cen_mom_g[ibin]-np.abs(cen_mom_g[ibin]*sys_cen_mom_g[ibin]/100), alpha=0.2,color='pink')
        # ax0.legend(loc='upper left',fontsize=16,ncol=2)    

        # plot_folder = "../scripts_perlmutter/plots_central_moment"
        # if not os.path.exists(plot_folder):
        #     os.makedirs(plot_folder)
        # fig.savefig(os.path.join(plot_folder,"central_{}_moment_{}.{}".format(order,flags.niter,flags.img_fmt)))

