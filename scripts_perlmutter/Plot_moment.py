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

parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/jing/', help='Folder containing data and MC files')
parser.add_argument('--weights', default='/pscratch/sd/j/jing/H1PCT/weights_saved/', help='Folder to store trained weights')

parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_4d_general.json', help='Basic config file containing general options')
parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')

parser.add_argument('--cf', default='c', help='Do central jets or forward')
parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--upTo', type=int, default=6, help='Highest order of the moment to compute to')
parser.add_argument('--clip',type=float,default=100, help='quantile to clip large trained weights')

parser.add_argument('--fast', action='store_true', default=False,help='Debugging with one bin and only central region')
parser.add_argument('--mom', action='store_true', default=False,help='Compute and output the moments')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')


flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)

mc_names = ['Pythia_nominal']#,'Sherpa_Lund']
# standalone_predictions = ['Herwig','Sherpa']
standalone_predictions = []    
data_idx = 0 # sample that after weights represent data
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1] # MC ref is used to define the reference simulation used to derive the closure and model systematic uncertainties
print("data_name: ", data_name)
print("mc_ref: ", mc_ref)

version = data_name


def RatioLabel(ax1,var):
    ax1.set_ylabel('Rel. diff. [%]')
    ax1.set_xlabel(r'$p_T$ [GeV]')    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    # ax1.axhline(y=10, color='r', linestyle='--')
    # ax1.axhline(y=-10, color='r', linestyle='--')
    ylim = [-20,20]
    ax1.set_ylim(ylim)


def compute_moment(values, weights, order):

    # first normalize the weights
    norm_wgt = weights / np.sum( weights, axis=0 )

    # raise event level value to a power, then add weights, then sum up (integrate dirac delta ftn)
    data_moment = np.sum( np.multiply( np.power(values, order), norm_wgt ), axis=0)

    return data_moment


def computeBinnedMoment(hist, order, pT):

    # to compute the nth moment in i_pt bin
    # sum over binContent * binCenter**order 
    value = []

    if pT < 9:
        binCenters = [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9]

        for i, binCenter in enumerate(binCenters):
            if i == 0 or i == 7:
                value.append( 0.2* hist[i] * binCenter**order )  # [0, 0.2] or [0.8, 1]
                # if i == 7: print("bin [0.8, 1]: ", 0.2* hist[i] * binCenter**order)
            else:
                value.append( 0.1* hist[i] * binCenter**order )

    else:  # [800, 1200, 2500]
        binCenters = [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.85]

        for i, binCenter in enumerate(binCenters):
            if i == 0:
                value.append( 0.2* hist[i] * binCenter**order )  # [0, 0.2] 
            elif i == 6: 
                value.append( 0.3* hist[i] * binCenter**order )  # [0.7, 1.0]
                # print("bin [0.7, 1]: ", 0.3* hist[i] * binCenter**order)
            else:
                value.append( 0.1* hist[i] * binCenter**order )

    print("!!!!! values in each bin: ", value)

    return sum(value)


def CalcK1sFromMoments(T1s):
    k1s = []
    for idx,mom in enumerate(T1s):
        k1s.append(T1s[idx])
    return k1s

def CalcK2sFromMoments(T1s, T2s):
    k2s = []
    for idx,mom in enumerate(T1s):
        k2s.append(T2s[idx]
                   - T1s[idx]**2)
    return k2s

def CalcK3sFromMoments(T1s, T2s, T3s):
    k3s = []
    for idx,mom in enumerate(T1s):
        k3s.append(T3s[idx]
                   - 3*T2s[idx]*T1s[idx]
                   + 2*T1s[idx]**3)
    return k3s

def CalcK4sFromMoments(T1s, T2s, T3s, T4s):
    k4s = []
    for idx,mom in enumerate(T1s):
        k4s.append( T4s[idx]
                    - 4*T3s[idx]*T1s[idx]
                    - 3*T2s[idx]**2
                    + 12*T2s[idx]*T1s[idx]**2
                    - 6*T1s[idx]**4 )
    return k4s

def CalcK5sFromMoments(T1s, T2s, T3s, T4s, T5s):
    k5s = []
    for idx,mom in enumerate(T1s):
        k5s.append(T5s[idx]
                   - 5*T4s[idx]*T1s[idx]
                   - 10*T3s[idx]*T2s[idx]
                   + 20*T3s[idx]*T1s[idx]**2
                   + 30*(T2s[idx]**2)*T1s[idx]
                   - 60*T2s[idx]*T1s[idx]**3 + 24*T1s[idx]**5 )
    return k5s

def CalcK6sFromMoments(T1s, T2s, T3s, T4s, T5s, T6s):
    k6s = []
    for idx,mom in enumerate(T1s):
        k6s.append(T6s[idx]
                   - 6*T5s[idx]*T1s[idx]
                   - 15*T4s[idx]*T2s[idx]+30*T4s[idx]*(T1s[idx]**2)
                   - 10*(T3s[idx]**2)+120*T3s[idx]*T2s[idx]*T1s[idx]
                   - 120*T3s[idx]*(T1s[idx]**3)
                   + 30*(T2s[idx]**3)
                   - 270*(T2s[idx]**2)*(T1s[idx]**2)
                   + 360*T2s[idx]*(T1s[idx]**4)
                   - 120*(T1s[idx]**6) )
    return k6s


def LoadData(q2_int):
    mc_info = {}
    weights_data = {}
    clips_data = {}
    sys_variations = {}
    sys_clips = {}
    print('q2_int: ', q2_int)
    # gen_q2 (pT) bin:  [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]
    # loading the bin:  gen_q2[q2_int-1] < pT < gen_q2[q2_int]
    # for q2_int in range(1,len(xaxis)+1)

    # actually full Run2
    mc_info['data'] = MCInfo('Data1516',flags.N,flags.data_folder,config,flags.cf,q2_int,is_reco=True,use_fiducial_mask=True)  

    # Loading saved weights 
    for mc_name in mc_names:
        print("{}.h5".format(mc_name))    
        mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder,config,flags.cf,q2_int,is_reco=flags.plot_reco,use_fiducial_mask=True)

        if mc_name == data_name: # Pythia_nominal
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)
            # weights_data[flags.mode] = mc_info[mc_name].ReturnWeights(flags.niter,model_name=model_name,mode=flags.mode)

            weights_data[flags.mode] = mc_info[mc_name].LoadTrainedWeights(
                                os.path.join(flags.weights,'nominal_iter{}.h5'.format(flags.niter)) #,flags.clip
                            )
            print("{}.h5".format(mc_name), weights_data[flags.mode].shape[0])

            if flags.sys: # load systematic weights

                for unc in opt.sys_sources:
                    if unc in ['model','closure','stat','QED']: continue
                    model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        flags.weights,base_name,version.replace("Pythia_nominal",unc),flags.niter)
                    print(mc_name.replace("Pythia_nominal",unc))
                    mc_info[unc] = MCInfo(mc_name.replace("Pythia_nominal",unc),int(flags.N),flags.data_folder,config,flags.cf,q2_int,is_reco=flags.plot_reco,use_fiducial_mask=True)
                    # sys_variations[unc] = mc_info[unc].ReturnWeights(
                    #     flags.niter,model_name=model_name,mode=flags.mode)
                    sys_variations[unc] = mc_info[unc].LoadTrainedWeights(
                                os.path.join(flags.weights,'{}_iter{}.h5'.format(unc, flags.niter)) #,flags.clip
                            )
                    print(mc_name.replace("Pythia_nominal",unc), sys_variations[unc].shape[0])
                    
                if not flags.plot_reco:
                    #Load non-closure weights
                    # model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                    #     flags.weights,base_name,version+'_closure',flags.niter)
                    # # sys_variations['closure'] = mc_info[mc_name].ReturnWeights(
                    # #     flags.niter,model_name=model_name,mode=flags.mode)
                    # sys_variations['closure'] = mc_info[mc_name].LoadTrainedWeights(
                    #             os.path.join(flags.weights,'closure_iter{}.h5'.format(flags.niter))
                    #         )
                
                    sys_variations['stat'] = []
                    # sys_variations['stat'] = [
                    #     mc_info[mc_name].LoadTrainedWeights(
                    #         os.path.join(flags.data_folder,'weights','{}_{}.h5'.format(mc_name,nstrap))
                    #     ) for nstrap in range(1,config['NBOOTSTRAP']+1)]
                else:
                    sys_variations['stat'] = []
                    
        # elif flags.sys and not flags.plot_reco:
        #     base_name = "Omnifold_{}".format(flags.mode)
        #     model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,mc_name,flags.niter)
        #     sys_variations['model'] = mc_info[mc_name].ReturnWeights(
        #         flags.niter,model_name=model_name,mode=flags.mode)
            
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

bin_mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
bin_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
py_cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

bin_sys_mom_cf = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
# bin_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 

# cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 
# py_cen_mom_qg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict))) 


###
orders = range(1,flags.upTo+1)
 
# pT binning: [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2500]
gen_var_names = opt.gen_vars
binning = opt.dedicated_binning['gen_pt_c']

if flags.fast == False:
    xaxis = 0.5*(binning[:-1] + binning[1:])
else:
    # for faster debugging
    xaxis = [250, 350]#, 450, 550, 650, 750, 850, 950, 1100, 1300]#, 1500, 2050]

print("binning", binning, "x axis", xaxis)
print("xaxis shape: ", len(xaxis))


for var in gen_var_names:
    if flags.cf == 'c':
        if not ( 'tf_c' in var ): continue
    elif flags.cf == 'f':
        if not ( 'tf_f' in var ): continue
    print(var)

    stat_unc = []
    sys_unc = []

    binning = opt.dedicated_binning[var]

    # Loop over pT first so just loop the array once and compute for all orders
    for q2_int in range(1,len(xaxis)+1):  # len of the pT bin list

        mc_info,weight_data,sys_variations = LoadData(q2_int)

        pT = xaxis[q2_int-1] # pT bin center, corresponding to the q/g fraction dict
        print("pT: ", pT)

        # For smooth moments
        data_var = mc_info[data_name].LoadVar(var)
        print("data_var dim: ", data_var.shape[0])

        mc_name = mc_names[0] # do only Pythia
        mc_var = mc_info[mc_name].LoadVar(var)

        # print("var count: ", data_var.shape)
        # print("mc info data weights: ", mc_info[data_name].nominal_wgts.shape)
        # print("weight data shape: ", weight_data.shape)
        # input()

        # Binned unfolded data
        # Gen level Pythia + OmniFold wgts -> data predictions
        data_pred,_=np.histogram(data_var,weights=weight_data*mc_info[data_name].nominal_wgts,bins=binning,density=True)
        print("hist: ", data_pred)
        # input()

        if flags.mom:
            
            for order in orders:

                # smooth moments
                data_moment = compute_moment(data_var,weights=(weight_data*mc_info[data_name].nominal_wgts),order=order)
                # data_moment = weighted_avg_and_std(data_var,weights=(weight_data*mc_info[data_name].nominal_wgts))[order-1]
                mom_cf[pT][var]['nominal'][order] = data_moment

                mc_moment = compute_moment(mc_var,weights=mc_info[mc_name].nominal_wgts,order=order)
                # mc_moment = weighted_avg_and_std(mc_var,weights=mc_info[mc_name].nominal_wgts)[order-1]
                py_mom_cf[pT][var]['nominal'][order] = mc_moment

                # binned moments
                bin_moment = computeBinnedMoment(data_pred, order, q2_int)
                bin_mom_cf[pT][var]['nominal'][order] = bin_moment

                print("moment check: ", data_moment, bin_moment)


        #######################################
        # Processing systematic uncertainties #
        #######################################
        if flags.sys == True:
            for order in orders:

                data_moment = mom_cf[pT][var]['nominal'][order]
                bin_moment = bin_mom_cf[pT][var]['nominal'][order]

                # stat_unc.append(0)
                # if not flags.plot_reco:
                # #Stat uncertainty
                #     straps = []
                #     for strap in range(len(sys_variations['stat'])):
                #         sys_pred = compute_moment(data_var,weights=sys_variations['stat'][strap]*mc_info[data_name].nominal_wgts,order=order)
                #         straps.append(sys_pred)
                #     stat_unc.append(np.std(straps,axis=0)/np.mean(straps,axis=0))
                # else:
                #     stat_unc.append(np.sqrt(data_var.shape[0])/data_var.shape[0])

                total_sys= 0 # stat_unc[-1]**2  

                for unc in opt.sys_sources:

                    # sys_wgt_sum = np.sum(mc_info[unc].nominal_wgts)
                    # sys_prescale = np.divide( np.sqrt(np.sum( np.power( mc_info['unc'].nominal_wgts, 2))), sys_wgt_sum)

                    if unc == 'stat': continue
                    if unc == 'QED': continue
                    if unc == 'closure':continue
                    elif unc == 'model':
                        #Model uncertainty: difference between unfolded values
                        data_sys = mc_info[mc_ref].LoadVar(var)
                        mask_var = np.abs(data_sys)>=0
                        # sys_moment,sys_std = weighted_avg_and_std(data_sys[mask_var],weights=(sys_variations[unc]*mc_info[unc].nominal_wgts)[mask_var])
                        sys_moment = compute_moment(data_sys,weights=(sys_variations[unc]*mc_info[unc].nominal_wgts),order=order)
                        ratio_sys = 100*np.divide(sys_moment-data_moment, data_moment)
                        # ratio_sys= np.sqrt(np.abs(ratio_sys**2 - stat_unc[-1]**2))
                        print("ratio_sys: ", ratio_sys)
                    else:
                        data_sys = mc_info[unc].LoadVar(var)
                        mask_var = np.abs(data_sys)>=0

                        data_pred_sys,_=np.histogram(data_sys,weights=(sys_variations[unc]*mc_info[unc].nominal_wgts),bins=binning,density=True)
                        print("sys hist: ", data_sys)
                        # if flags.plot_reco:
                        #     sys_moment = np.average(data_sys,weights=mc_info[unc].nominal_wgts)
                        #     mc_var = mc_info[data_name].LoadVar(var)
                        #     pred,_=np.average(mc_var,weights=mc_info[data_name].nominal_wgts)
                        #     ratio_sys = 100*np.divide(sys_moment-pred,pred)
                        # else:
                        # sys_moment,sys_std = weighted_avg_and_std(data_sys[mask_var],weights=(sys_variations[unc]*mc_info[unc].nominal_wgts)[mask_var])
                        sys_moment = compute_moment(data_sys,weights=(sys_variations[unc]*mc_info[unc].nominal_wgts),order=order)

                        bin_sys_moment = computeBinnedMoment(data_pred_sys, order, q2_int)
                        bin_sys_mom_cf[pT][var]['sys'][order] = bin_moment

                        print("nominal: ", data_moment, "sys: ", sys_moment, "binned sys: ", bin_sys_moment)

                        # ratio_sys = 100*np.divide(sys_moment-data_moment, data_moment)
                        # print("ratio_sys: ", ratio_sys)
                        sys_corr = sys_moment/bin_sys_moment
                        nom_corr = data_moment/bin_moment
                        binning_unc = sys_corr - nom_corr
                        frac_binning_unc = np.divide(sys_moment/bin_sys_moment - data_moment/bin_moment, data_moment/bin_moment)
                        print("raw sys corr: ", sys_corr, "raw nominal corr: ", nom_corr)
                        print("binning correction uncertainties: ", binning_unc, "frac: ", frac_binning_unc)

                    # total_sys+= np.abs(ratio_sys**2)# - stat_unc[-1]**2)
                    # print(ratio_sys**2,unc)

                #total_sys = np.abs(total_sys - stat_unc[-1]**2)
                # total_sys = total_sys**0.5 
                # print("order: ", order, "total sys: ", total_sys)
                # mom_cf[pT][var]['sys'][order] = total_sys

        else:
            sys_unc.append(0)
            # stat_unc.append(0)


    # end loop over pT bins

    t1s, t2s, t3s, t4s, t5s, t6s = [], [], [], [], [], []

    for q2_int in range(1,len(xaxis)+1):  # len of the pT bin list

        pT = xaxis[q2_int-1] # pT bin center, corresponding to the q/g fraction dict
        print("pT: ", pT)

        t1s.append( mom_cf[pT][var]['nominal'][1] )
        t2s.append( mom_cf[pT][var]['nominal'][2] )
        t3s.append( mom_cf[pT][var]['nominal'][3] )
        t4s.append( mom_cf[pT][var]['nominal'][4] )
        t5s.append( mom_cf[pT][var]['nominal'][5] )
        t6s.append( mom_cf[pT][var]['nominal'][6] )

    k1s = CalcK1sFromMoments( t1s ) 
    k2s = CalcK2sFromMoments( t1s, t2s )
    k3s = CalcK3sFromMoments( t1s, t2s, t3s )
    k4s = CalcK4sFromMoments( t1s, t2s, t3s, t4s )
    k5s = CalcK5sFromMoments( t1s, t2s, t3s, t4s, t5s )
    k6s = CalcK6sFromMoments( t1s, t2s, t3s, t4s, t5s, t6s )

    k2k2s = np.power( k2s, 2)
    k2k3s = np.multiply( k2s, k3s)
    k2k2k2s = np.power( k2s, 3)

    ks = [k2k2s, k2k3s, k2k2k2s, k4s, k5s, k6s]

    # Binned version:

    bin_t1s, bin_t2s, bin_t3s, bin_t4s, bin_t5s, bin_t6s = [], [], [], [], [], []

    for q2_int in range(1,len(xaxis)+1):  # len of the pT bin list

        pT = xaxis[q2_int-1] # pT bin center, corresponding to the q/g fraction dict
        print("pT: ", pT)

        bin_t1s.append( bin_mom_cf[pT][var]['nominal'][1] )
        bin_t2s.append( bin_mom_cf[pT][var]['nominal'][2] )
        bin_t3s.append( bin_mom_cf[pT][var]['nominal'][3] )
        bin_t4s.append( bin_mom_cf[pT][var]['nominal'][4] )
        bin_t5s.append( bin_mom_cf[pT][var]['nominal'][5] )
        bin_t6s.append( bin_mom_cf[pT][var]['nominal'][6] )

    bin_k1s = CalcK1sFromMoments( bin_t1s ) 
    bin_k2s = CalcK2sFromMoments( bin_t1s, bin_t2s )
    bin_k3s = CalcK3sFromMoments( bin_t1s, bin_t2s, bin_t3s )
    bin_k4s = CalcK4sFromMoments( bin_t1s, bin_t2s, bin_t3s, bin_t4s )
    bin_k5s = CalcK5sFromMoments( bin_t1s, bin_t2s, bin_t3s, bin_t4s, bin_t5s )
    bin_k6s = CalcK6sFromMoments( bin_t1s, bin_t2s, bin_t3s, bin_t4s, bin_t5s, bin_t6s )

    bin_k2k2s = np.power( bin_k2s, 2)
    bin_k2k3s = np.multiply( bin_k2s, bin_k3s)
    bin_k2k2k2s = np.power( bin_k2s, 3)

    bin_ks = [bin_k2k2s, bin_k2k3s, bin_k2k2k2s, bin_k4s, bin_k5s, bin_k6s]

    print("!!!!!k4_k2k2: ", k4s, k2k2s, "\n")
    print("!!!!!k5_k2k3: ", k5s, k2k3s, "\n")
    print("!!!!!k6_k2k2k2: ", k6s, k2k2k2s, "\n")

    print("!!!!!k4_k2k2: ", bin_k4s, bin_k2k2s, "\n")
    print("!!!!!k5_k2k3: ", bin_k5s, bin_k2k3s, "\n")
    print("!!!!!k6_k2k2k2: ", bin_k6s, bin_k2k2k2s, "\n")


    if flags.mom:
        # Outputting the moments for each variable, one array / order
        for order in orders:
            data_pred = []
            py_pred = []
            bin_pred = []
            sys_unc = []

            for q2_int in range(1,len(xaxis)+1):
                pT = xaxis[q2_int-1]
                data_pred.append( mom_cf[pT][var]['nominal'][order] )
                py_pred.append( py_mom_cf[pT][var]['nominal'][order] )
                bin_pred.append( bin_mom_cf[pT][var]['nominal'][order] )
                if flags.sys == True:
                    sys_unc.append( mom_cf[pT][var]['sys'][order] )

            data_pred = np.array( data_pred )
            py_pred = np.array( py_pred )
            bin_pred = np.array( bin_pred )
            if flags.sys == True:
                sys_unc = np.array( sys_unc )
                print("sys: ", sys_unc)

            # toSave = np.zeros( (len(binning)-1, 5))
            print(data_pred.shape )
            toSave = np.zeros( (len(xaxis), 7))
            toSave[:, 0] = xaxis
            toSave[:, 1] = data_pred
            toSave[:, 2] = py_pred
            toSave[:, 3] = bin_pred
            toSave[:, 4] = np.array( ks[order-1] )
            toSave[:, 5] = np.array( bin_ks[order-1] )
            
            f = h5.File(var+"_"+str(order)+'.h5', 'w')
            f['xaxis']=toSave[:,0]
            f[var]=toSave[:, 1]
            f['py_'+var]=toSave[:, 2]
            f['bin_'+var]=toSave[:, 3]
            f['cen_'+var]=toSave[:, 4]
            f['bin_cen_'+var]=toSave[:, 5]

            if flags.sys == True:
                toSave[:, 6] = sys_unc
                f['sys_'+var]=toSave[:, 6]

            f.close()  
