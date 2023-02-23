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

parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('--plot_flow', action='store_true', default=False,help='Plot the RGflow mixing')
parser.add_argument('--plot_mom', action='store_true', default=False,help='Plot the moments')
parser.add_argument('--plot_dist', action='store_true', default=False,help='Plot the overlaid distributions')
parser.add_argument('--plot_n', action='store_true', default=False,help='Plot the event / bin count')

parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--q2_int', type=int, default=0, help='Q2 interval to consider')
parser.add_argument('--img_fmt', default='png', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config = LoadJson(flags.config)

folder = 'results'
plot_folder = '/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/plots_mom'
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
# binning = np.linspace(0,1,50)
binning = opt.dedicated_binning['gen_tf_c']
xaxis = 0.5*(binning[:-1] + binning[1:])
# print(xaxis)

pT_binning = opt.dedicated_binning['pt_c']
pT_x = 0.5*(pT_binning[:-1] + pT_binning[1:])
mc = 'Pythia'

if flags.plot_reco == True:
    gen_vars = ['tf_c','tf_f']
elif flags.plot_flow == True:
    gen_vars = ['central_mom']
elif flags.plot_mom:
    gen_vars = ['gen_tf_g','gen_tf_q']
elif flags.plot_n == True:
    gen_vars = ['reco_tf_c','reco_data_tf_c']
    # gen_vars = ['gen_tf_c','data_gen_tf_c']

for var in gen_vars:

    ###########################

    if flags.plot_mom == True:
        mom_q, mom_g = {}, {}
        py_mom_q, py_mom_g = {}, {}
        # ax0.set_ylim(top=3)
        key = 'gen_tf'

        # for each order
        for order in range(1,5):
            print(os.path.join(data_folder,"{}_{}.h5".format(var,order)))
            # input()
            f = h5.File(os.path.join(data_folder,"{}_{}.h5".format(var,order)),'r')
            pT_xaxis = np.array(f['xaxis'][:])
            data_pred = np.array(f[var][:])
            py_pred = np.array(f['py_'+var][:])
            ratio = 100*np.divide(py_pred-data_pred,data_pred)
            print(data_pred, py_pred, ratio)
            input()

            fig,gs = opt.SetGrid(2) 
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1],sharex=ax0)
            ax1.set_ylim(top=1.1, bottom=0.9)
            # ax0.set_xscale('log')
            opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$moment order $'+str(order), ax0=ax0)
            #ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
            ax0.set_ylim(top=1,bottom = 0)
            ax0.set_xlim(left=130,right=1.2e3)
            # ax0.legend(loc='upper left',fontsize=16,ncol=2)  
            text_ypos = 0.85 #text position height
            text_xpos = 0.22      
                
            plt.text(text_xpos, text_ypos,
                     '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = ax0.transAxes, fontsize=18)

            if flags.sys == True:
                print("x axis: ", pT_xaxis, "data size: ", len(data_pred), "err: ", np.abs(data_pred)*sys_unc/100.0 )
                # sqrt of sys already taken when writing into the sys_dict above #L262
                ax0.errorbar(pT_xaxis, data_pred, yerr = np.abs(data_pred)*sys_unc/100, fmt='o', ms=12, color='k', label='Data') 
            else:
                mc = 'Pythia'
                ax0.plot(pT_xaxis,py_pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
                ax0.plot(pT_xaxis,data_pred, marker='o', lw=0, ms=12, color='k',label='Data')
                ax1.plot(pT_xaxis,ratio,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

            fig.savefig(os.path.join(plot_folder,"moment_{}_{}.{}".format(var,order,flags.img_fmt)))


    if flags.plot_flow == True:
        # data_folder = '/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/central_mom'
        cen_mom_q, cen_mom_g = {}, {}
        py_cen_mom_q, py_cen_mom_g = {}, {}
        # ax0.set_ylim(top=3)
        # ax1 = plt.subplot(gs[1],sharex=ax0)
        # ax1.set_ylim(top=1.1, bottom=0.9)
        key = 'cen_mom'

        # for each order
        for order in range(1,7):
            print(os.path.join(data_folder,"{}_{}.h5".format(var,order)))
            # input()
            f = h5.File(os.path.join(data_folder,"{}_{}.h5".format(var,order)),'r')
            cen_mom_q[order] = f[key+'_q'][:8]
            cen_mom_g[order] = f[key+'_g'][:8]
            py_cen_mom_q[order] = f['py_'+key+'_q'][:8]
            py_cen_mom_g[order] = f['py_'+key+'_g'][:8]
            # pT = cen_mom_q[order] = f['xaxis'][:]

        for i in [4,5,6]:
            if i == 4:
                fig,gs = opt.SetGrid(npanels) 
                ax0 = plt.subplot(gs[0])
                ax0.plot(py_cen_mom_q[2]**2, py_cen_mom_q[4],color='skyblue',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='quark, k_4, Pythia')
                ax0.plot(py_cen_mom_g[2]**2, py_cen_mom_g[4],color='pink',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_4, Pythia')
                ax0.plot(cen_mom_q[2]**2, cen_mom_q[4],color='b',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='quark, k_4, Data')
                ax0.plot(cen_mom_g[2]**2, cen_mom_g[4],color='r',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_4, Data')

                ax0.legend(loc='best',fontsize=10, ncol=2)
                # frag
                ax0.set_xlim([-0.0007,0.0035])
                ax0.set_ylim([-0.002,0.0007])
                # # ntuples
                # ax0.set_xlim([-0.001,0.02])
                # ax0.set_ylim([-0.025,0.01])
                fig.savefig(os.path.join(plot_folder,"central_mom_RG_flow_k4_k2sq.{}".format(flags.img_fmt)))

            elif i == 5:
                fig,gs = opt.SetGrid(npanels) 
                ax0 = plt.subplot(gs[0])
                # j = [0,1,2,3,4,6,7,8,9,10,11,12,13]
                ax0.plot(py_cen_mom_q[3]*py_cen_mom_q[2], py_cen_mom_q[5],color='skyblue',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='quark, k_5, Pythia')
                ax0.plot(py_cen_mom_g[3]*py_cen_mom_g[2], py_cen_mom_g[5],color='pink',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_5, Pythia')
                ax0.plot(cen_mom_q[3]*cen_mom_q[2], cen_mom_q[5],color='b',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='quark, k_5, Data')
                ax0.plot(cen_mom_g[3]*cen_mom_g[2], cen_mom_g[5],color='r',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_5, Data')

                ax0.legend(loc='best',fontsize=10, ncol=2)
                ax0.set_xlim([-0.00025,0.00004])
                ax0.set_ylim([-0.0003,0.00085])
                fig.savefig(os.path.join(plot_folder,"central_mom_RG_flow_k5_k2k3.{}".format(flags.img_fmt)))
            
            elif i == 6:
                fig,gs = opt.SetGrid(npanels) 
                ax0 = plt.subplot(gs[0])
                ax0.plot(py_cen_mom_q[2]**3, py_cen_mom_q[6],color='skyblue',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='quark, k_6, Pythia')
                ax0.plot(py_cen_mom_g[2]**3, py_cen_mom_g[6],color='pink',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_6, Pythia')
                ax0.plot(cen_mom_q[2]**3, cen_mom_q[6],color='b',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='quark, k_6, Data')
                ax0.plot(cen_mom_g[2]**3, cen_mom_g[6],color='r',marker='*',ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label='gluon, k_6, Data')

                ax0.legend(loc='best',fontsize=10, ncol=2)
                ax0.set_xlim([-0.00005,0.00017])
                ax0.set_ylim([-0.00018,0.00025])
                fig.savefig(os.path.join(plot_folder,"central_mom_RG_flow_k6_k2k2k2.{}".format(flags.img_fmt)))


    # for full TF distribution
    if flags.plot_dist == True:
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


    if flags.plot_n == True:
        print(var)

        lowestBin, evt = [], []

        for i in range(1,len(pT_binning)):

            print(os.path.join(data_folder,"{}_{}.h5".format(var,i)))
            f = h5.File(os.path.join(data_folder,"{}_{}.h5".format(var,i)),'r')

            pT_xaxis = np.array(f['xaxis'][:])
            lowestBin.append(f['tf_c'][0]) # the lowest bin count
            evt.append( f['num'][0] )
            # ratio = 100*np.divide(py_pred-data_pred,data_pred)
        
        lowestBin = np.array(lowestBin)
        evt = np.array(evt)
        lowestBinCount = np.multiply(lowestBin, evt)
        
        print(lowestBin, evt, lowestBinCount)
        # input()

        for i,toPlot in enumerate([lowestBin, evt, lowestBinCount]):
            if i == 0: 
                fig,gs = opt.SetGrid(1) 
                ax0 = plt.subplot(gs[0])
                ax0.grid(True)
                # ax0.set_xscale('log')
                opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$pT binned count$', ax0=ax0)
                ax0.set_ylim(top=1.5*max(toPlot),bottom = 0)
                ax0.set_xlim(left=130,right=2.5e3)
                ax0.legend(loc='best',fontsize=16,ncol=2)
                    
                plt.text(text_xpos, text_ypos,
                         '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax0.transAxes, fontsize=18)
                plt.title('"lowestBin_fraction"')

                ax0.step(pT_x, toPlot, color='g', where='mid', label='Lowest_bin_fraction_'+var)
                fig.show()
                input()
                fig.savefig(os.path.join(plot_folder,"lowestBin_fraction_{}.{}".format(var,flags.img_fmt)))
            elif i == 1:
                fig,gs = opt.SetGrid(1) 
                ax0 = plt.subplot(gs[0])
                ax0.grid(True)
                opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$pT binned count$', ax0=ax0)
                ax0.set_ylim(top=1.5*max(toPlot),bottom = 0.5*min(toPlot))
                ax0.set_yscale('log')
                ax0.set_xlim(left=130,right=2.5e3)
                ax0.legend(loc='best',fontsize=16,ncol=2) 
                text_ypos = 0.85 #text position height
                text_xpos = 0.22      
                    
                plt.text(text_xpos, text_ypos,
                         '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax0.transAxes, fontsize=18)
                plt.title('"total_jet_count"')

                ax0.step(pT_x, toPlot,color='g',where='mid',label='event_count_'+var)
                fig.savefig(os.path.join(plot_folder,"event_count_{}.{}".format(var,flags.img_fmt)))
            else:
                fig,gs = opt.SetGrid(1) 
                ax0 = plt.subplot(gs[0])
                ax0.grid(True)
                opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$pT binned count$', ax0=ax0)
                ax0.set_ylim(top=1.5*max(toPlot),bottom = 0.5*min(toPlot))
                ax0.set_xlim(left=130,right=2.5e3)
                ax0.set_yscale('log')
                ax0.legend(loc='best',fontsize=16,ncol=2) 
                text_ypos = 0.85 #text position height
                text_xpos = 0.22      
                    
                plt.text(text_xpos, text_ypos,
                         '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV \n$p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax0.transAxes, fontsize=18)
                plt.title('"lowestBin_jet_count"')

                ax0.step(pT_x, lowestBinCount, color='b', where='mid', label='Lowest content_'+var)
                fig.savefig(os.path.join(plot_folder,"lowestBin_count_{}.{}".format(var,flags.img_fmt)))
