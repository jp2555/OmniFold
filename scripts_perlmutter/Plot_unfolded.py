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

parser.add_argument('--data_folder', default='/pscratch/sd/j/jing/h5', help='Folder containing data and MC files')
parser.add_argument('--weights', default='/pscratch/sd/j/jing/H1PCT/weights_saved', help='Folder to store trained weights')
parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_6d_general.json', help='Basic config file containing general options')

parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--saveArrays', action='store_true', default=False,help='output arrays for quick plotting')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('-N',type=float,default=300e6, help='Number of events to evaluate')
parser.add_argument('--clip',type=float,default=100, help='quantile to clip large trained weights')

parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
parser.add_argument('--nom', default='trial1', help='version of nominal result...')
parser.add_argument('--q2_int', type=int, default=0, help='Q2 interval to consider')
parser.add_argument('--img_fmt', default='png', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config = LoadJson(flags.config)

if 'closure' in opt.sys_sources.keys() or flags.closure:
    mc_names = ['Pythia_nominal','Sherpa_Lund']
else: 
    mc_names = ['Pythia_nominal']

# standalone_predictions = ['Herwig','Sherpa']
standalone_predictions = []

data_idx = 0 #Sample that after weights represent data
data_name = mc_names[data_idx] # Pythia  
mc_ref = mc_names[data_idx-1] #MC ref is used to define the reference simulation used to derive the closure and model systematic uncertainties

print("data: ", data_name, "MC ref: ", mc_ref)

folder = 'results'
text_ypos = 0.7 #text position height
text_xpos = 0.82
version = data_name
if flags.closure:
    version  +='_closure'

if flags.plot_reco:
    gen_var_names = opt.reco_vars
else:
    gen_var_names = opt.gen_vars


def RatioLabel(ax1,var):
    ax1.set_ylabel('Rel. diff. [%]')
    ax1.set_xlabel(gen_var_names[var])    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    # ax1.axhline(y=10, color='r', linestyle='--')
    # ax1.axhline(y=-10, color='r', linestyle='--')
    ylim = [-70,70]
    if 'ncharged' in var:
        ylim = [-100,150]

    if flags.plot_reco or flags.closure:
        ylim = [-50,50]
        
    ax1.set_ylim(ylim)
    

def PlotUnc(xaxis,var,values,xlabel='',add_text=''):
    fig = plt.figure(figsize=(9, 9))
    ax = plt.gca()
    opt.FormatFig(xlabel = xlabel, ylabel = 'Uncertainty',ax0=ax)
    total_sys = np.zeros(len(xaxis))
    max_y, min_y = [], []
    sys_name = {
            'Pythia_TRACK2':'Incl. Eff.',
            'Pythia_TRACK1':'Saggita Bias',
            'Pythia_TRACK3':'CTIDE',
            'Pythia_TRACK4':'Fake Rate',
            'Pythia_JES45':'Composition up',
            'Pythia_JES46':'Composition down',
            'Pythia_JES47':'Response up',
            'Pythia_JES48':'Response down',
            }
    for sys in values:
        if sys == 'stat':
            #print(values[sys], np.shape(values[sys]) )
            #plt.step(xaxis, values[sys],where='mid',color=opt.sys_sources[sys],label=sys)
            continue
        if sys == 'ensem':
            #plt.step(xaxis, values[sys],where='mid',color=opt.sys_sources[sys],label=sys)
            continue
        else:
            # plt.hist(np.abs(values[sys]),color=opt.sys_sources[sys],label=sys)#opt.sys_translate[sys])
            # plt.hlines(values[sys],xmin=xmin,xmax=xmax,color=opt.sys_sources[sys],label=sys)#opt.sys_translate[sys])
            if 'TRACK' in sys or 'JES' in sys:
                plt.step(xaxis, values[sys]/100,where='mid',color=opt.sys_sources[sys],label=sys_name[sys])#opt.sys_translate[sys])
            else:
                plt.step(xaxis, values[sys]/100,where='mid',color=opt.sys_sources[sys],label=sys)
        total_sys+=values[sys]**2
        max_y.append(np.max(values[sys]))
        min_y.append(np.min(values[sys]))

    total_sys= total_sys**0.5
    plt.step(xaxis, total_sys/100,where='mid',color='k',label='total')  
    # plt.hist(np.sqrt(total_sys),ls=":",color='black',histtype='step',label='Total Syst.',lw=3)
    # max_y = np.max(np.sqrt(total_sys))
    ref = np.zeros(1)
    xmin,xmax = xaxis[0], xaxis[-1]
    #plt.hlines(total_sys,xmin=xmin,xmax=xmax,linestyles='dashed',color='k')
    plt.hlines(ref,xmin=xmin,xmax=xmax,linestyles='dashed',color='grey')#opt.sys_translate[sys])
    #max_ylim, min_ylim = 1,0
    # max_ylim, min_ylim = np.max(max_y), np.min(min_y)

    if len(add_text)>0:
        #ax = plt.gca()
        #opt.WriteText(text_xpos,text_ypos,add_text,ax)
        plt.text(text_xpos, text_ypos,add_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax.transAxes, fontsize=18)
        
    plt.legend(loc='lower right',fontsize=12,ncol=2)
    if 'pt' in var: 
        plt.ylim([-0.2,0.2])
    if sys == 'stat':
        plt.ylim([0,0.01])
    else:
        plt.ylim([-0.12,0.12])
    return fig



mc_info = {}
weights_data = {}
clips_data = {}

sys_variations = {}
sys_clip = {}

# if flags.closure:
#     mc_info['data'] = MCInfo(data_name,flags.N,flags.data_folder,config,flags.q2_int,is_reco=True) 
# else:
mc_info['data'] = MCInfo('Data1516',flags.N,flags.data_folder,config,flags.q2_int,is_reco=True)   
#Loading weights from training
for mc_name in mc_names:
    print("{}.h5".format(mc_name))    
    mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder,config,flags.q2_int,is_reco=flags.plot_reco)
    if mc_name == data_name:
        if flags.comp:
            for mode in config['COMPARISON']:
                print(mode)
                base_name = "Omnifold_{}".format(mode)
                model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)
                weights_data[mode] = mc_info[mc_name].ReturnWeights(config['{}_NITER'.format(mode)],model_name=model_name,mode=mode)
        else:
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)


            # weights_data[flags.mode] = mc_info[mc_name].ReturnWeights(flags.niter,model_name=model_name,mode=flags.mode)
            weights_data[flags.mode], clips_data[flags.mode] = mc_info[mc_name].LoadTrainedWeights(
                os.path.join(flags.weights,'nominal_iter{}_ver_{}.h5'.format(flags.niter,flags.nom)),flags.clip 
                )
            print(weights_data[flags.mode].shape[0])


            if flags.sys == True: #load systematic variations
                for sys in opt.sys_sources:
                    if sys in ['model','closure','stat','QED','ensem']: continue
                    model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        flags.weights,base_name,version.replace("Pythia_nominal",sys),flags.niter)

                    # mc_info[sys] = MCInfo(mc_name.replace("Pythia_nominal",opt.sys_translate[sys]),int(flags.N),flags.data_folder,config,flags.q2_int,is_reco=flags.plot_reco)
                    mc_info[sys] = MCInfo(mc_name.replace("Pythia_nominal",sys),int(flags.N),flags.data_folder,config,flags.q2_int,is_reco=flags.plot_reco)
                    # sys_variations[sys] = mc_info[sys].ReturnWeights(
                    #     flags.niter,model_name=model_name,mode=flags.mode)
                    sys_variations[sys],sys_clip[sys] = mc_info[sys].LoadTrainedWeights( # first evaluate weights, here define a clip mask
                                os.path.join(flags.weights,'{}_iter{}.h5'.format(sys, flags.niter)),flags.clip
                            )
                    print(mc_name.replace("Pythia_nominal",sys), sys_variations[sys].shape[0])

                    
                if not flags.plot_reco: 
                    if 'closure' in opt.sys_sources:
                        #Load non-closure weights
                        # model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        #     flags.weights,base_name,version+'_closure',flags.niter)
                        # sys_variations['closure'] = mc_info[mc_name].ReturnWeights(
                        #     flags.niter,model_name=model_name,mode=flags.mode)
                        print("iteration: ", flags.niter)
                        sys_variations['closure'] = mc_info[mc_name].LoadTrainedWeights(
                                os.path.join(flags.weights,'closure_iter{}.h5'.format(flags.niter)),flags.clip
                            ) 
                        sys_variations['stat'] = []
                        sys_variations['ensem'] = []

                    elif 'stat' in opt.sys_sources:
                        sys_variations['ensem'] = []
                        sys_variations['stat'] = [
                            mc_info[mc_name].LoadTrainedWeights(
                                os.path.join(flags.weights,'{}_{}.h5'.format(mc_name,nstrap)),flags.clip
                                # os.path.join(flags.data_folder,'{}_{}.h5'.format(mc_name,nstrap))
                            ) for nstrap in range(1,config['NBOOTSTRAP']+1)]
                        # print(sys_variations['stat'])

                    elif 'ensem' in opt.sys_sources:
                        sys_variations['stat'] = []
                        sys_variations['ensem'] = [
                            mc_info[mc_name].LoadTrainedWeights(
                                os.path.join(flags.weights,'{}_trial{}.h5'.format(mc_name,ntrial)),flags.clip
                                # os.path.join(flags.data_folder,'{}_{}.h5'.format(mc_name,ntrial))
                            ) for ntrial in range(1,20)]
                        # print(sys_variations['ensem'])
                    else:
                        sys_variations['stat'] = []
                        sys_variations['ensem'] = []
                else:
                    sys_variations['stat'] = []
                    sys_variations['ensem'] = []
                    
    #elif flags.sys and not flags.plot_reco:
    #    base_name = "Omnifold_{}".format(flags.mode)
    #    model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,mc_name,flags.niter)
    #    sys_variations['model'] = mc_info[mc_name].ReturnWeights(
    #        flags.niter,model_name=model_name,mode=flags.mode)
            
weight_data = weights_data[flags.mode]
clip_data = clips_data[flags.mode]


for var in gen_var_names:
    print(var)
    if not flags.plot_reco: var_truth = opt.var_truth_translate[var] # to correspond to the truth hist names

    binning = opt.dedicated_binning[var]
    if flags.plot_reco:
        data_var = mc_info['data'].LoadVar(var,clip_data)
    else:
        data_var = mc_info[data_name].LoadVar(var,clip_data)

    npanels = 3
    if flags.plot_reco or flags.closure:
        npanels=2
    fig,gs = opt.SetGrid(npanels) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex=ax0)
    if flags.plot_reco == False and flags.closure==False:
        ax2 = plt.subplot(gs[2],sharex=ax0)
    #xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    xaxis = 0.5*(binning[:-1] + binning[1:])
    
    # if 'gen_pt_c' in var or 'gen_pt_f' in var or 'gen_trkpt_f' in var or 'gen_trkpt_f' in var:
    #     ax0.set_xscale('log')
    #     ax0.set_yscale('log')
    #     ax1.set_xscale('log')

    print("var count: ", data_var.shape)
    print("mc info data weights: ", mc_info[data_name].nominal_wgts[clip_data].shape)
    print("weight data shape: ", weight_data.shape)
    # input()

    if flags.plot_reco:
        data_pred,_=np.histogram(data_var,weights=mc_info['data'].nominal_wgts,bins=binning,density=True)
    elif not( 'closure' in opt.sys_sources):
        data_pred,_=np.histogram(data_var,weights=weight_data*mc_info[data_name].nominal_wgts[clip_data],bins=binning,density=True)

    if 'tf_c' in var and flags.saveArrays == True:# or 'tf_f' in var:
        toSave = np.zeros( (len(binning)-1, 2))
        toSave[:, 0] = xaxis
        toSave[:, 1] = data_pred
        df = h5.File("data_"+var+"_"+str(flags.q2_int)+'.h5', 'w')
        df['xaxis']=toSave[:,0]
        df[var]=toSave[:, 1]
        df.close()  
 
    ratios = {}
    if flags.q2_int==0 and flags.plot_reco==False:
        ax0.set_ylim(top=opt.fixed_yaxis[var])
        #max_y = max(data_pred)
        #ax0.set_ylim(top=1.5*max_y)
    else:
        max_y = max(data_pred)
        if 'tf_c' in var or 'tf_f' in var:
            ax0.set_ylim(top=3)
        else:
            ax0.set_ylim(top=1.5*max_y)

    #######################################
    # Processing systematic uncertainties #
    #######################################
    if flags.sys == True:
        ratio_sys = {}
        total_sys = np.ones(len(binning)-1)

        if not flags.plot_reco: var_truth = opt.var_truth_translate[var]

        if not flags.plot_reco:
            #Stat uncertainty
            stat_unc = []
            #print( len(sys_variations['stat']))
            #input()
            for strap in range(len(sys_variations['stat'])):
                sys_pred,_ = np.histogram(data_var,weights=sys_variations['stat'][strap]*mc_info[data_name].nominal_wgts,bins=binning,density=True)
                stat_unc.append(sys_pred)

            ratio_sys['stat'] = 100*np.std(stat_unc,axis=0)/np.mean(stat_unc,axis=0)
            print(ratio_sys['stat'])

            if '_c' in var and flags.saveArrays == True:
                toSave_stat = np.zeros( (len(binning)-1, 2))
                toSave_stat[:, 0] = xaxis
                toSave_stat[:, 1] = ratio_sys['stat']
                df = h5.File("stat_"+var+"_"+str(flags.q2_int)+'.h5', 'w')
                df['xaxis']=toSave_stat[:,0]
                df[var]=toSave_stat[:, 1]
                df.close()  
                # input()

            #Ensemble variance
            ensem_unc = []
            #print( len(sys_variations['ensem']))
            #input()
            #for trial in range(len(sys_variations['ensem'])):
            #    sys_pred,_ = np.histogram(data_var,weights=sys_variations['ensem'][trial]*mc_info[data_name].nominal_wgts,bins=binning,density=True)
            #    ensem_unc.append(sys_pred)

            #ratio_sys['ensem'] = 100*np.std(ensem_unc,axis=0)/np.mean(ensem_unc,axis=0) # axis=0 to compute the statistical property for each bin
           # ratio_sys['ensem'] = 100*(np.max( np.abs(ensem_unc-np.mean(ensem_unc,axis=0)),axis=0)  )/np.mean(ensem_unc,axis=0)
            #print(ratio_sys['ensem'])
        else:
            counts,_ = np.histogram(data_var,bins=binning)
            ratio_sys['stat'] = np.sqrt(counts)/counts*100
            
        total_sys+= ratio_sys['stat']**2
        #Plot data and stat error
        # print(xaxis,data_pred,np.abs(binning),data_pred*ratio_sys['stat']/100.)
        # ax0.errorbar(xaxis,data_pred,yerr = data_pred*ratio_sys['stat']/100.,fmt='o',ms=12,color='k',label='Data')
        # for ibin in range(len(xaxis)):
        #     xup = binning[ibin+1]
        #     xlow = binning[ibin]
        #     # yup=data_pred[ibin]*(1 + ratio_sys['stat'][ibin]/100.)
        #     # ylow=data_pred[ibin]*(1 - ratio_sys['stat'][ibin]/100.)
        #     yup=data_pred[ibin]*(1 )
        #     ylow=data_pred[ibin]*(1)
        #     if ibin==0:
        #         ax0.fill_between(np.array([xlow,xup]),yup,ylow, alpha=0.3,color='k',label='Total unc.')
        #     else:
        #         ax0.fill_between(np.array([xlow,xup]),yup,ylow, alpha=0.3,color='k')
                
        for sys in opt.sys_sources:
            if sys == 'stat' or sys == 'ensem': continue
            if flags.plot_reco == True:
                if sys in ['QED','model','closure']: continue 

            if sys == 'QED':
                qed_unc=LoadJson('QED_sys.json')
                ratio_sys[sys] = 100*np.array(qed_unc[var][str(flags.q2_int)])                
            elif sys == 'model':
                #Model uncertainty: difference between unfolded values
                data_sys = mc_info[mc_ref].LoadVar(var)
                sys_pred,_ = np.histogram(data_sys,weights=sys_variations[sys]*mc_info[mc_ref].nominal_wgts,bins=binning,density=True)
                ratio_sys[sys] = 100*np.divide(sys_pred-data_pred,data_pred)
                # ratio_sys[sys] = np.sqrt(np.abs(ratio_sys[sys]**2 - ratio_sys['stat']**2))
            elif sys == 'closure':
                sys_pred,_ = np.histogram(data_var,weights=sys_variations[sys]*mc_info[data_name].nominal_wgts,bins=binning,density=True)
                mc_var = mc_info[mc_ref].LoadVar(var)
                mc_pred,_ = np.histogram(mc_var,weights=mc_info[mc_ref].nominal_wgts,bins=binning,density=True)
                ratio_sys[sys] = 100*np.divide(sys_pred-mc_pred,mc_pred)
                # ratio_sys[sys] = np.sqrt(np.abs(ratio_sys[sys]**2 - ratio_sys['stat']**2))
            else:
                data_sys = mc_info[sys].LoadVar(var,sys_clip[sys]) # a clip mask defined above when loading trained wgts for this sys 
                print("sys var count: ", data_sys.shape)
                print("mc info sys weights: ", mc_info[sys].nominal_wgts[sys_clip[sys]].shape)
                # input()
                if flags.plot_reco:
                    sys_pred,_ = np.histogram(data_sys,weights=mc_info[sys].nominal_wgts[sys_clip[sys]],bins=binning,density=True)
                else:
                    sys_pred,_ = np.histogram(data_sys,weights=sys_variations[sys]*mc_info[sys].nominal_wgts[sys_clip[sys]],bins=binning,density=True)
                
                if not flags.plot_reco:
                    ratio_sys[sys] = 100*np.divide(sys_pred-data_pred,data_pred)
                    # print("err at L305",data_pred,ratio_sys[sys])
                    # ratio_sys[sys] = np.sqrt(np.abs(ratio_sys[sys]**2 - ratio_sys['stat']**2))
                else:
                    mc_var = mc_info[data_name].LoadVar(var)
                    pred,_=np.histogram(mc_var,weights=mc_info[data_name].nominal_wgts,
                                        bins=binning,density=True)
                    ratio_sys[sys] = 100*np.divide(sys_pred-pred,pred)
                total_sys+= ratio_sys[sys]**2

            if flags.saveArrays == True:
                toSave = np.zeros( (len(binning)-1, 2))
                toSave[:, 0] = xaxis
                toSave[:, 1] = ratio_sys[sys]
                df = h5.File(var+"_"+sys+"_"+str(flags.q2_int)+'.h5', 'w')
                df['xaxis']=toSave[:,0]
                df[var]=toSave[:, 1]
                df.close() 

        
        for ibin in range(len(xaxis)):
            xup = binning[ibin+1]
            xlow = binning[ibin]
            ax1.fill_between(np.array([xlow,xup]),np.sqrt(total_sys[ibin]),-np.sqrt(total_sys[ibin]), alpha=0.3,color='k')            
            if flags.plot_reco == False and flags.closure==False:
                ax2.fill_between(np.array([xlow,xup]),np.sqrt(total_sys[ibin]),-np.sqrt(total_sys[ibin]), alpha=0.3,color='k')
    else:
        label_data = 'Data' if flags.closure == False else 'Pythia weighted'
        if flags.plot_reco:
            ax0.errorbar(xaxis,data_pred,yerr = np.sqrt(counts)/(sum(counts) * np.diff(binning)),fmt='o',ms=12,color='k',label=label_data)
            data_pred,_,_=ax0.hist(data_var,bins=binning,weights=mc_info['data'].nominal_wgts,label=label_data,
                                   density=True,color="black",histtype="step")

            counts,_ = np.histogram(data_var,bins=binning)
            for ibin in range(len(xaxis)):
                xup = binning[ibin+1]
                xlow = binning[ibin]
                ax1.fill_between(np.array([xlow,xup]),
                                 np.sqrt(counts[ibin])/counts[ibin]*100,
                                 -np.sqrt(counts[ibin])/counts[ibin]*100, alpha=0.3,color='k')

        else:
            data_pred,_,_=ax0.hist(data_var,weights=weight_data*mc_info[data_name].nominal_wgts,bins=binning,label=label_data,density=True,color="black",histtype="step")
    
    var_name = gen_var_names[var]
    if 'ncharge' in var:
        var_name = r'$\tilde{\lambda}_0^0$'
    elif 'charge' in var:
        var_name = r'$\tilde{\lambda}_0^1$'
    elif 'ptD' in var:
        var_name = r'$\sqrt{\lambda_0^2}$'
        
    if flags.plot_reco:
        opt.FormatFig(xlabel = "", ylabel = r'1/N $\mathrm{dN}/\mathrm{d}$%s'%var_name,ax0=ax0)
    else:
        opt.FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%var_name,ax0=ax0)
        
    add_text = ''
    if flags.plot_reco:
        add_text = 'Detector level events \n'
        text_yshift = 0.03
    else:
        text_yshift = 0
    if flags.q2_int>0:
        gen_q2 = opt.dedicated_binning['gen_pt_c']        
        text_q2 = "${} < p_T < {} ~GeV^2$ \n".format(int(np.round(gen_q2[flags.q2_int-1],1)),int(np.round(gen_q2[flags.q2_int],1)))
        text_fiducial = add_text+text_q2+'$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV  \n $p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$'
        # opt.WriteText(xpos=0.27,ypos=1.03,text = text_q2,ax0=ax0)

        
        plt.text(text_xpos, text_ypos+text_yshift,text_fiducial,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax0.transAxes, fontsize=16)
        
    else:
        text_fiducial = '$\eta<2.1$ \n $p_\mathrm{T}^\mathrm{jet}>160$ GeV  \n $p_\mathrm{T}^\mathrm{leading}/p_\mathrm{T}^\mathrm{subleading}<1.5$'
        plt.text(text_xpos, text_ypos,add_text+text_fiducial,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax0.transAxes, fontsize=16)
        
    ax0.tick_params(axis='x',labelsize=0)


    ################################
    # Processing other predictions #
    ################################

    for i,mc_name in enumerate(mc_names):
        #Upper canvas
        mc = mc_name.split("_")[0]
        mc_var = mc_info[mc_name].LoadVar(var,clip_data)

        pred,_=np.histogram(mc_var,weights=mc_info[mc_name].nominal_wgts[clip_data],bins=binning,density=True)
        # print( "Pythia predictions: ", pred)

        if 'tf_c' in var and flags.saveArrays == True:# or 'tf_f' in var:
            toSave = np.zeros( (len(binning)-1, 2))
            toSave[:, 0] = xaxis
            toSave[:, 1] = pred
            f = h5.File(var+"_"+str(flags.q2_int)+'.h5', 'w')
            f['xaxis']=toSave[:,0]
            f[var]=toSave[:, 1]
            f.close()  

        # we need at reco level hist to normalize it & compare to our weighted data
        if (flags.plot_reco):
            pred,_,_=ax0.hist(mc_var,weights=mc_info[mc_name].nominal_wgts,bins=binning,label=mc,
                                   density=True,color=opt.colors[mc],histtype="step")
        else: # plot truth
            ax0.plot(xaxis,pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)

        ratios[mc] = 100*np.divide(pred-data_pred,data_pred)        
        # print("ratio at L397",pred)#,ratio[mc])
        # Ratio plot
        displacement = opt.xaxis_disp[mc]
        xpos = xaxis+np.abs(np.diff(binning)/2.)*displacement
        # ax1.plot(xpos,ratios[mc],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

    if flags.closure ==False and flags.plot_reco==False:
        for mc_name in standalone_predictions:
            #process additional theory samples already stored as histograms
            pred= opt.LoadFromROOT(os.path.join("../truth","{}.root".format(mc_name)),var_truth,flags.q2_int)
            if len(pred)>len(xaxis):
                pred=pred[:len(xaxis)] #In case less bins are used
            pred = pred / (np.sum(pred) * np.diff(binning))
            
            displacement = opt.xaxis_disp[mc_name]
            xpos = xaxis+np.abs(np.diff(binning)/2.)*displacement
            ax0.plot(xpos,pred,color=opt.colors[mc_name],marker=opt.markers[mc_name],
                     ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=opt.name_translate[mc_name])
            ratios[mc_name] = 100*np.divide(pred-data_pred,data_pred)
            #Ratio plot    
            ax2.plot(xpos,ratios[mc_name],color=opt.colors[mc_name],marker=opt.markers[mc_name],
                     ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    
    handles, labels = ax0.get_legend_handles_labels()
    order = [len(labels)-1] + list(range(0,len(labels)-1))
    ax0.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left',fontsize=16,ncol=2)
    ax0.legend()
    RatioLabel(ax1,var)
    if flags.plot_reco == False and flags.closure==False:
        ax1.tick_params(axis='x',labelsize=0)
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        RatioLabel(ax2,var)
        ax2.yaxis.set_label_coords(-0.08, 2)
    
    plot_folder = '../plots_'+data_name
    if flags.closure:
        plot_folder+='_closure'
    if flags.sys:
        plot_folder+='_sys'#+opt.sys_sources[0]
    if flags.plot_reco:
        plot_folder+='_reco'
    plot_folder+='_{}'.format(flags.mode)
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    if flags.q2_int>0:
        fig.savefig(os.path.join(plot_folder,"{}_{}_{}.{}".format(var,flags.niter,flags.q2_int,flags.img_fmt)))
        if flags.sys:
            fig_sys = PlotUnc(xaxis,var,ratio_sys,gen_var_names[var],text_fiducial)
            fig_sys.savefig(os.path.join(plot_folder,"sys_{}_{}_{}.{}".format(var,flags.niter,flags.q2_int,flags.img_fmt)))
    else:
        fig.savefig(os.path.join(plot_folder,"{}_{}.{}".format(var,flags.niter,flags.img_fmt)))
        if flags.sys:
            fig_sys = PlotUnc(xaxis,var,ratio_sys,gen_var_names[var],text_fiducial)
            fig_sys.savefig(os.path.join(plot_folder,"sys_{}_{}_{}_nom{}.{}".format(var,flags.niter,flags.clip,flags.nom, flags.img_fmt)))


    # if flags.comp:
    #     #Compare the non-closure uncertainty
    #     fig,gs = opt.SetGrid() 
    #     ax0 = plt.subplot(gs[0])
    #     ax0.tick_params(axis='x',labelsize=0)
        
    #     ax1 = plt.subplot(gs[1],sharex=ax0)

    #     if 'genjet_pt' in var or 'Q2' in var:
    #         ax0.set_xscale('log')
    #         ax0.set_yscale('log')
    #         ax1.set_xscale('log')
        
    #     mc_var = mc_info[mc_ref].LoadVar(var)        
    #     mc_pred,_,_=ax0.hist(mc_var,weights=mc_info[mc_ref].nominal_wgts,bins=binning,label="Target Gen",density=True,color="black",histtype="step")

    #     opt.FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var],ax0=ax0)
    #     if flags.q2_int>0:
    #         gen_q2 = opt.dedicated_binning['gen_pt_c']      
    #         text = r"{} < Q$^2$ < {} GeV$^2$".format(int(np.round(gen_q2[flags.q2_int-1],1)),int(np.round(gen_q2[flags.q2_int],1)))
    #         opt.WriteText(xpos=0.1,ypos=1.03,text = text,ax0=ax0)

    #     ratios = {}
        
    #     data_var = mc_info[data_name].LoadVar(var)
    #     for train in weights_data:
    #         pred,_,_ = ax0.hist(data_var,weights=weights_data[train]*mc_info[data_name].nominal_wgts,bins=binning,
    #                             label=train,density=True,color=opt.colors[train],histtype="step")
    #         ratios[train] = 100*np.divide(mc_pred-pred,pred)
            
    #         ax1.plot(xaxis,ratios[train],color=opt.colors[train],marker=opt.markers[train],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

            
    #     RatioLabel(ax1)
    #     ax0.legend(loc='lower right',fontsize=16,ncol=1)

    #     plot_folder = '../plots_comp_'+data_name
    #     if not os.path.exists(plot_folder):
    #         os.makedirs(plot_folder)


    #     if flags.q2_int>0:
    #         fig.savefig(os.path.join(plot_folder,"{}_{}_{}.{}".format(var,flags.niter,flags.q2_int,flags.img_fmt)))
    #     else:
    #         fig.savefig(os.path.join(plot_folder,"{}_{}.{}".format(var,flags.niter,flags.img_fmt)))







