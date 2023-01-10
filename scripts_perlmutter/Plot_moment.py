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
parser.add_argument('--weights', default='/pscratch/sd/j/jing/H1PCT/weights_1516', help='Folder to store trained weights')
parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')

parser.add_argument('--fast', action='store_true', default=False,help='Debugging with one bin and only central region')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--qg', action='store_true', default=False,help='Do q/g splitting or not')
parser.add_argument('--cf', action='store_true', default='c',help='Do central jets or forward')
parser.add_argument('--flow', action='store_true', default=False,help='Process cumulants for RG flow mixing plots')

parser.add_argument('-N',type=float,default=100e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=3, help='Omnifold iteration to load')
parser.add_argument('--upTo', type=int, default=5, help='Highest order of the moment to compute to')


flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)

mc_names = ['Pythia_nominal']
# mc_names = ['Rapgap_nominal','Djangoh_nominal']
# standalone_predictions = ['Herwig','Sherpa']
standalone_predictions = []    
data_idx = 0 #Sample that after weights represent data
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1] #MC ref is used to define the reference simulation used to derive the closure and model systematic uncertainties
print(mc_ref)

version = data_name


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
    variance = np.average((values)**2, weights=weights)
    return (average, variance)


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


def LoadData(q2_int):
    mc_info = {}
    weights_data = {}
    sys_variations = {}
    # gen_q2 (pT) bin:  [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]
    # loading the bin:  gen_q2[q2_int-1] < pT < gen_q2[q2_int]
    # for q2_int in range(1,len(xaxis)+1)

    mc_info['data'] = MCInfo('Data1516',flags.N,flags.data_folder,config,q2_int,is_reco=True)  
    #Loading weights from training
    for mc_name in mc_names:
        print("{}.h5".format(mc_name))    
        mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder,config,q2_int,is_reco=flags.plot_reco)
        if mc_name == data_name: # Pythia_nominal
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)

            weights_data[flags.mode] = mc_info[mc_name].ReturnWeights(flags.niter,model_name=model_name,mode=flags.mode)

            if flags.sys == True: #load systematic variations
                for unc in opt.sys_sources:
                    if unc in ['model','closure','stat','QED']: continue
                    model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        flags.weights,base_name,version.replace("Pythia_nominal",unc),flags.niter)
                    print(mc_name.replace("Pythia_nominal",unc))
                    mc_info[unc] = MCInfo(mc_name.replace("Pythia_nominal",unc),int(flags.N),flags.data_folder,config,q2_int,is_reco=flags.plot_reco)
                    sys_variations[unc] = mc_info[unc].ReturnWeights(
                        flags.niter,model_name=model_name,mode=flags.mode)
                    
                # if not flags.plot_reco:
                #     #Load non-closure weights
                #     model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                #         flags.weights,base_name,version+'_closure',flags.niter)
                #     sys_variations['closure'] = mc_info[mc_name].ReturnWeights(
                #         flags.niter,model_name=model_name,mode=flags.mode)
                
                #     sys_variations['stat'] = [
                #         mc_info[mc_name].LoadTrainedWeights(
                #             os.path.join(flags.data_folder,'weights','{}_{}.h5'.format(mc_name,nstrap))
                #         ) for nstrap in range(1,config['NBOOTSTRAP']+1)]
                # else:
                sys_variations['stat'] = []
                    
        elif flags.sys and not flags.plot_reco:
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,mc_name,flags.niter)
            sys_variations['model'] = mc_info[mc_name].ReturnWeights(
                flags.niter,model_name=model_name,mode=flags.mode)
            
    weight_data = weights_data[flags.mode]
    return mc_info,weight_data,sys_variations


################################################
# Now processing the samples
################################################


# easy multi-key dict
mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

py_mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
py_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
py_cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

orders = range(1,flags.upTo+1)
 
# pT binning: [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]
if flags.plot_reco:
    gen_var_names = opt.reco_vars
    binning = opt.dedicated_binning['pt_c']
else:
    gen_var_names = opt.gen_vars
    binning = opt.dedicated_binning['gen_pt_c']

if flags.fast == False:
    xaxis = 0.5*(binning[:-1] + binning[1:])
else:
# for faster debugging
    xaxis = [250, ]#350, 450, 550, 650, 750, 850, 950, 1100, 1300, 1500, 1800, 2250]

print("binning", binning, "x axis", xaxis)
print("xaxis shape: ", len(xaxis))


for var in gen_var_names:
    if flags.cf == 'c':
        if not ( 'tf_c' in var ): continue
    elif flags.cf == 'f':
        if not ( 'tf_f' in var ): continue
    else:
        if not ( 'tf_c' in var or 'tf_f' in var ): continue
    print(var)

    data_pred = []
    sys_unc = []
    stat_unc = []

    for q2_int in range(1,len(xaxis)+1):  # len of the pT bin list

        mc_info,weight_data,sys_variations = LoadData(q2_int)
        pT = xaxis[q2_int-1] # pT bin center, corresponding to the q/g fraction dict
        print("pT: ", pT)

        if flags.plot_reco:
            data_var = mc_info['data'].LoadVar(var)
        else:
            data_var = mc_info[data_name].LoadVar(var)
            # do only Pythia
        
        mc_name = mc_names[0]
        mc_var = mc_info[mc_name].LoadVar(var)
        # mask_var = np.abs(data_var)>=0

        for order in orders:

            # if order == 2: # cross check with numpy function
            #     mc_moment = weighted_avg_and_std(mc_var,weights=mc_info[mc_name].nominal_wgts)[1]
            # else:
            if flags.plot_reco:
                data_moment = compute_moment(data_var,weights=mc_info['data'].nominal_wgts,order=order)
                # data_moment = weighted_avg_and_std(data_var,weights=mc_info['data'].nominal_wgts)[order-1]
            else:
                data_moment = compute_moment(data_var,weights=(weight_data*mc_info[data_name].nominal_wgts),order=order)
                # data_moment = weighted_avg_and_std(data_var,weights=(weight_data*mc_info[data_name].nominal_wgts))[order-1]
            mom_cf[pT][var]['nominal'][order] = data_moment

            mc_moment = compute_moment(mc_var,weights=mc_info[mc_name].nominal_wgts,order=order)
            # mc_moment = weighted_avg_and_std(mc_var,weights=mc_info[mc_name].nominal_wgts)[order-1]
            py_mom_cf[pT][var]['nominal'][order] = mc_moment

        # print("data moment :", mom_cf)
        # print("Pythia moment:", py_mom_cf)
        # input()

        #######################################
        # Processing systematic uncertainties #
        #######################################
        if flags.sys == True:
            for order in orders:
                data_moment = mom_cf[pT][var]['nominal'][order]
                # stat_unc.append(0)
                # if not flags.plot_reco:
                # #Stat uncertainty
                #     straps = []
                #     for strap in range(len(sys_variations['stat'])):
                #         sys_pred = weighted_moments(data_var,weights=sys_variations['stat'][strap]*mc_info[data_name].nominal_wgts,mom=flags.mom)
                #         straps.append(sys_pred)
                #     stat_unc.append(np.std(straps,axis=0)/np.mean(straps,axis=0))
                # else:
                #     stat_unc.append(np.sqrt(data_var.shape[0])/data_var.shape[0])

                total_sys= 0 # stat_unc[-1]**2  
                for unc in opt.sys_sources:
                    if unc == 'stat': continue
                    if unc == 'QED': continue
                    if unc == 'closure':continue
                    elif unc == 'model':
                        #Model uncertainty: difference between unfolded values
                        data_sys = mc_info[mc_ref].LoadVar(var)
                        mask_var = np.abs(data_sys)>=0
                        # sys_moment,sys_std = weighted_avg_and_std(data_sys[mask_var],weights=(sys_variations[unc]*mc_info[mc_ref].nominal_wgts)[mask_var])
                        sys_moment = compute_moment(data_sys,weights=(sys_variations[unc]*mc_info[mc_ref].nominal_wgts),order=order)
                        ratio_sys = 100*np.divide(sys_moment-data_moment, data_moment)
                        # ratio_sys= np.sqrt(np.abs(ratio_sys**2 - stat_unc[-1]**2))
                        print("ratio_sys: ", ratio_sys)
                    else:
                        data_sys = mc_info[unc].LoadVar(var)
                        mask_var = np.abs(data_sys)>=0
                        if flags.plot_reco:
                            sys_moment = np.average(data_sys,weights=mc_info[unc].nominal_wgts)
                            mc_var = mc_info[data_name].LoadVar(var)
                            pred,_=np.average(mc_var,weights=mc_info[data_name].nominal_wgts)
                            ratio_sys = 100*np.divide(sys_moment-pred,pred)
                        else:
                            # sys_moment,sys_std = weighted_avg_and_std(data_sys[mask_var],weights=(sys_variations[unc]*mc_info[unc].nominal_wgts)[mask_var])
                            sys_moment = compute_moment(data_sys,weights=(sys_variations[unc]*mc_info[unc].nominal_wgts),order=order)
                            print("nominal: ", data_moment, "sys: ", sys_moment)
                            ratio_sys = 100*np.divide(sys_moment-data_moment, data_moment)
                            print("ratio_sys: ", ratio_sys)

                    total_sys+= np.abs(ratio_sys**2)# - stat_unc[-1]**2)
                    # print(ratio_sys**2,unc)
                #total_sys = np.abs(total_sys - stat_unc[-1]**2)
                total_sys = total_sys**0.5 # taking sqrt when plotting later
                print("order: ", order, "total sys: ", total_sys)
                mom_cf[pT][var]['sys'][order] = total_sys
                # sys_unc.append(total_sys)
        else:
            sys_unc.append(0)
            # stat_unc.append(0)


################################################
# Do q/g splitting
################################################
if flags.qg == True:
    coefs = {
    # GLUON fraction i.e (f_cg, f_fg)
    # central (large), forward (small)
    # from fragmentation measurement 
    # 250: (66.2, 56.7), 350: (62.1, 50.8),
    # 450: (59.3, 46.3), 550: (56.2, 42.3),
    # 650: (52.9, 39.4), 750: (50.5, 36.9),
    # 850: (47.6, 34.7), 950: (45.2, 32.8),
    # 1100: (42.01, 30.50), 1300: (37.55, 27.48), 
    # 1500: (34.10, 25.18), 1800: (30.07, 21.73), 
    # 2250: (23.31, 17.41),
    # }
    # Fron Ntuples
    250: (64.86, 60.11), 350: (61.14, 55.39),
    450: (57.62, 51.12), 550: (54.36, 47.66),
    650: (51.44, 44.74), 750: (48.71, 42.25),
    850: (46.05, 40.02), 950: (43.70, 38.11),
    1100: (40.71, 35.88), 1300: (36.64, 32.72), 
    1500: (32.92, 30.07), 1800: (28.85, 27.11), 
    2250: (22.95, 21.99),
    }


    if flags.sys == True: 
        variations = ['nominal','sys']
    else:
        variations = ['nominal']

    for order in orders:
        for q2_int in range(1,len(xaxis)+1):
            pT_coef = xaxis[q2_int-1] # here fore reading q/g fraction, same as the pT key
            print(pT_coef, 'nominal')

            f_cg, f_fg = coefs[pT_coef][0]*0.01, coefs[pT_coef][1]*0.01
            f_cq, f_fq = 1-f_cg, 1-f_fg
            # print("f_cg, f_fg, f_cq, f_fq: ", f_cg, f_fg, f_cq, f_fq)

            for variation in variations:

                # print( "Tc", mom_cf[pT_coef]['gen_tf_c'][variation])
                Tc = mom_cf[pT_coef]['gen_tf_c'][variation][order]
                Tf = mom_cf[pT_coef]['gen_tf_f'][variation][order]
                Tqg = qg_splitting(f_cq, f_cg, f_fq, f_fg, Tc, Tf) 
                # print(Tqg)
               
                mom_qg[variation][order]['gen_tf_q'][pT_coef] = Tqg[0] #np.abs(Tqg[0])
                mom_qg[variation][order]['gen_tf_g'][pT_coef] = Tqg[1] #np.abs(Tqg[1])

                py_Tc = py_mom_cf[pT_coef]['gen_tf_c']['nominal'][order]
                py_Tf = py_mom_cf[pT_coef]['gen_tf_f']['nominal'][order]
                py_Tqg = qg_splitting(f_cq, f_cg, f_fq, f_fg, py_Tc, py_Tf) 

                py_mom_qg['nominal'][order]['gen_tf_q'][pT_coef] = py_Tqg[0] #np.abs(py_Tqg[0])
                py_mom_qg['nominal'][order]['gen_tf_g'][pT_coef] = py_Tqg[1] #np.abs(py_Tqg[1])

    if flags.plot_reco:
        print("reco moment :", mom_qg)
        print("reco Pythia moment:", py_mom_qg)
    else:
        print("moment q/g:", mom_qg['nominal'])
        print("Pythia moment q/g:", py_mom_qg['nominal'])
    # input()

if flags.qg == True:
    flavors = ['gen_tf_q','gen_tf_g']
elif flags.fast == True:
    flavors = ['gen_tf_c']
else:
    flavors = ['gen_tf_c','gen_tf_f']

for flavor in flavors:

    for order in orders: # one list / order
        fig,gs = opt.SetGrid(1) 
        ax0 = plt.subplot(gs[0])
        # ax0.set_xscale('log')
        opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$track function moments',ax0=ax0)
        #ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
        ax0.set_xlim(left=130,right=2.5e3)
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

        for q2_int in range(1,len(xaxis)+1):
            pT = xaxis[q2_int-1]
            if flags.qg == True:
                data_pred.append( mom_qg['nominal'][order][flavor][pT] )
                py_pred.append( py_mom_qg['nominal'][order][flavor][pT] )
            else:
                # checking before q/g splitting
                data_pred.append( mom_cf[pT][flavor]['nominal'][order] )
                py_pred.append( py_mom_cf[pT][flavor]['nominal'][order] )
                # sys_unc.append( mom_qg['sys'][order][flavor][pT] )

        data_pred = np.array( data_pred )
        sys_unc = np.array( sys_unc )
        py_pred = np.array( py_pred )

        toSave = np.zeros( (len(binning)-1, 4))
        toSave[:, 0] = xaxis
        toSave[:, 1] = data_pred
        toSave[:, 2] = py_pred
        
        f = h5.File(flavor+"_"+str(order)+'.h5', 'w')
        f['xaxis']=toSave[:,0]
        f[flavor]=toSave[:, 1]
        f['py_'+flavor]=toSave[:, 2]

        if flags.sys == True:
            toSave[:, 3] = sys_unc
            f['sys_'+flavor]=toSave[:, 3]

        f.close()  

        # print(flavor, 'data: ', data_pred, 'pythia: ', py_pred)
        # xaxis = [250, 350, 450, 550, 650, 750, 850, 950, 1100, 1300]#, 1500, 1800, 2250]

        if flags.sys == True:
            print("x axis: ", xaxis, "data size: ", len(data_pred), "err: ", np.abs(data_pred)*sys_unc/100.0 )
            # sqrt of sys already taken when writing into the sys_dict above #L262
            ax0.errorbar(xaxis, data_pred, yerr = np.abs(data_pred)*sys_unc/100, fmt='o', ms=12, color='k', label='Data') 
        else:
            mc = 'Pythia'
            ax0.plot(xaxis,py_pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
            ax0.plot(xaxis,data_pred, marker='o', lw=0, ms=12, color='k',label='Data')

        # for ibin in range(len(xaxis)):
        #     xup = binning[ibin+1]
        #     xlow = binning[ibin] 
        #     ax0.hlines(y=data_pred[ibin], xmin=xlow, xmax=xup, colors='black')
        #     ax1.fill_between(np.array([xlow,xup]),np.sqrt(sys_unc[ibin]),-np.sqrt(sys_unc[ibin]), alpha=0.3,color='k')

        # for mc_name in mc_names:
        #     mc = mc_name.split("_")[0]
        #     ax0.plot(xaxis,mc_pred[mc_name],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
            # ratio = np.divide(np.array(mc_pred[mc_name])-data_pred,data_pred)
            #ax1.plot(xaxis,ratio,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
                           
        if flags.plot_reco: 
            plot_folder = '../plots_reco_'+data_name+"_"+flavor
        else: 
            plot_folder = '../plots_'+data_name+"_"+flavor
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"{}_{}_{}.{}".format(flavor,flags.niter,order,flags.img_fmt)))


################################################
# Processing cumulants for RG flow mixing plots
################################################

if flags.flow == True and flags.qg == True:
    for order in orders:
        cen_mom_q, cen_mom_g = [], []
        py_cen_mom_q, py_cen_mom_g = [], []

        for q2_int in range(1,len(xaxis)+1):
            pT_coef = xaxis[q2_int-1]
            Tq1, Tg1 = mom_qg['nominal'][1]['gen_tf_q'][pT_coef], mom_qg['nominal'][1]['gen_tf_g'][pT_coef]
            Tq2, Tg2 = mom_qg['nominal'][2]['gen_tf_q'][pT_coef], mom_qg['nominal'][2]['gen_tf_g'][pT_coef]
            Tq3, Tg3 = mom_qg['nominal'][3]['gen_tf_q'][pT_coef], mom_qg['nominal'][3]['gen_tf_g'][pT_coef]
            Tq4, Tg4 = mom_qg['nominal'][4]['gen_tf_q'][pT_coef], mom_qg['nominal'][4]['gen_tf_g'][pT_coef]
            Tq5, Tg5 = mom_qg['nominal'][5]['gen_tf_q'][pT_coef], mom_qg['nominal'][5]['gen_tf_g'][pT_coef]

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

            # for Pythia
            py_Tq1, py_Tg1 = py_mom_qg['nominal'][1]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][1]['gen_tf_g'][pT_coef]
            py_Tq2, py_Tg2 = py_mom_qg['nominal'][2]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][2]['gen_tf_g'][pT_coef]
            py_Tq3, py_Tg3 = py_mom_qg['nominal'][3]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][3]['gen_tf_g'][pT_coef]
            py_Tq4, py_Tg4 = py_mom_qg['nominal'][4]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][4]['gen_tf_g'][pT_coef]
            py_Tq5, py_Tg5 = py_mom_qg['nominal'][5]['gen_tf_q'][pT_coef], py_mom_qg['nominal'][5]['gen_tf_g'][pT_coef]

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

        cen_mom_q, cen_mom_g = np.array( cen_mom_q ), np.array( cen_mom_g )
        py_cen_mom_q, py_cen_mom_g = np.array( py_cen_mom_q ), np.array( py_cen_mom_g )

        toSave = np.zeros( (len(binning)-1, 5))
        toSave[:, 0] = xaxis
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

        # plotting for each order
        text_ypos = 0.85 #text position height
        text_xpos = 0.22

        fig,gs = opt.SetGrid(1) 
        ax0 = plt.subplot(gs[0])
        # ax0.tick_params(axis='x',labelsize=0)
        # ax0.set_xscale('log')
        opt.FormatFig(xlabel = r'$p_T$ [GeV]', ylabel = r'$p_\mathrm{T}^\mathrm{charged}/p_\mathrm{T}^\mathrm{all}$ central moments',ax0=ax0)
        # ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
        ax0.set_xlim(left=130,right=2.5e3)
        ax0.plot(xaxis,cen_mom_q,color='b',label='data_qurak')
        ax0.plot(xaxis,cen_mom_g,color='r',label='data_gluon')
        ax0.plot(xaxis,py_cen_mom_q,color='skyblue',label='Pythia_qurak')
        ax0.plot(xaxis,py_cen_mom_g,color='coral',label='Pythia_gluon')
        ax0.legend(loc='upper left',fontsize=16,ncol=2)    

        plot_folder = '../plots_'+data_name+"_"+"central_moment"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"central_{}_moment_{}.{}".format(order,flags.niter,flags.img_fmt)))
