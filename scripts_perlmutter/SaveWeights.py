import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import argparse
import os
import sys
import h5py as h5
from omnifold import  Multifold, Scaler, LoadJson
import tensorflow as tf
import tensorflow.keras.backend as K
sys.path.append('../')
import shared.options as opt
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


class MCInfo():
    def __init__(self,mc_name,N,data_folder,config,cf='f',q2_int=0,is_reco=False,use_fiducial_mask=True):
        self.N = int(N)
        self.file = h5.File(os.path.join(data_folder,"{}.h5".format(mc_name)),'r')
        self.is_reco = is_reco
        self.config = config
        self.pTbin = (use_fiducial_mask==False)
        # self.clip = (self.is_reco==False)

        if not self.is_reco:
            # self.truth_mask = self.file['pass_truth'][:self.N] #pass truth region definition
            self.truth_mask = 1
            if use_fiducial_mask:
                # particle_mask = np.sum(self.file['gen_jet_part_pt'][:self.N] > 0,-1) > 1
                # self.mask = (self.truth_mask==1)&(self.file['pass_fiducial'][:self.N] == 1) #pass fiducial region definition
                self.mask = (self.truth_mask==1)&(self.file['pass_truth'][:self.N] == 1) #pass fiducial region definition
                #self.mask = (self.mask) * (particle_mask)
            else:
                self.mask = (self.truth_mask==1)
            if cf == 'c':
                q2_name = 'gen_pt_c'
            elif cf == 'f':
                q2_name = 'gen_pt_f'
        else:
            self.mask = self.file['pass_reco'][:self.N] == 1 #pass fiducial region definition
            if cf == 'c':
                q2_name = 'pt_c'
            elif cf == 'f':
                q2_name = 'pt_f'
            
        if q2_int>0:  # INDEX of the q2 binning
            gen_q2 = opt.dedicated_binning[q2_name]
            # self.pTmask = ((self.file[q2_name][:self.N] > gen_q2[q2_int-1]) & (self.file[q2_name][:self.N] < gen_q2[q2_int]) & (self.file['pass_truth'][:self.N] == 1))
            print("binning in truth *",q2_name,"* pT")
            
            self.mask *= ((self.file[q2_name][:self.N] > gen_q2[q2_int-1]) & (self.file[q2_name][:self.N] < gen_q2[q2_int]))
            #self.fiducial_mask *= np.sum(self.file['gen_jet_part_pt'][:self.N] > 0,-1) < 20   

        self.nominal_wgts = self.file['wgt'][:self.N][self.mask]
            
    def LoadVar(self,var,clip):

        return_var = self.file[var][:self.N][self.mask][clip]
        
        if 'tau' in var:
            return_var = np.ma.log(return_var).filled(0)
            
        return return_var


    def ReturnWeights(self,niter,model_name,mode='hybrid'):
        mfold = self.LoadDataWeights(niter,mode)
        return self.Reweight(mfold,model_name)
    
    def LoadDataWeights(self,niter,mode='hybrid'):

        mfold = Multifold(
            mode=mode,
            nevts = self.N
        )
        
        
        if mode == 'PCT':
            var_names = self.config['VAR_PCT_GEN']
            global_names = self.config['GLOBAL_GEN']
            global_vars = np.concatenate([np.expand_dims(self.file[var][:self.N],-1) for var in global_names],-1)
            mean,std = Scaler(self.file,global_names)
            global_vars = (global_vars - mean)/std

        else:
            var_names = self.config['VAR_MLP_GEN']
            global_vars = np.array([[]])

        mfold.global_vars = {'reco':global_vars}
        data = np.concatenate([np.expand_dims(self.file[var][:self.N],-1) for var in var_names],-1)
        # if mode != 'PCT':
        #     tau_idx = [4,5,6] #CAUTION!!! IF THAT CHANGES REMEMBER TO CHANGE THIS LINE TOO
        #     for idx in tau_idx:
        #         data[:,idx] = np.ma.log(data[:,idx]).filled(0)
        
        if mode != "PCT":
            mean,std = Scaler(self.file,var_names)
            data=(data-mean)/std
            mfold.mc_gen = data
        else:
            mfold.mc_gen = [data,global_vars]
            
        mfold.PrepareModel()
        return mfold

    def Reweight(self,mfold,model_name):
        mfold.model2.load_weights(model_name)
        return mfold.reweight(mfold.mc_gen,mfold.model2)[self.mask]        

    def LoadTrainedWeights(self,file_name):
        h5file = h5.File(file_name,'r')
        weights = h5file['wgt'][:self.N]#[self.mask]
        upper = np.percentile( weights, 99.7, axis=0) # 99.7 for 3 sigma
        clip = (weights < upper)

        weights = weights[clip]

        # print(weights.shape)
        if self.pTbin:
            weights = weights[0,:][self.mask]
        return weights, clip
        
            


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/pscratch/sd/j/jing/h5', help='Folder containing data and MC files')
    parser.add_argument('--mode', default='standard', help='Which train type to load [hybrid/standard/PCT]')
    parser.add_argument('--config', default='config_6d_general.json', help='Basic config file containing general options')
    parser.add_argument('--out', default='/pscratch/sd/j/jing/H1PCT/weights_saved/', help='Folder to save the weights')
    parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
    parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
    parser.add_argument('--pt', action='store_true', default=False,help='weights saved to bin in pT?')
    parser.add_argument('--nom', action='store_true', default=False,help='Evaluate results with nominal')
    parser.add_argument('-N',type=float,default=700e6, help='Number of events to evaluate')
   
    flags = parser.parse_args()
    config=LoadJson(flags.config)
   
    ###################
    #bootstrap weights#
    ###################
    base_name = "Omnifold_{}".format(flags.mode)
    mc_names = ['Pythia_nominal']   
    if flags.sys == True:
        sys_names = opt.sys_sources
    flags.N = int(flags.N)
    if flags.pt: # for plotting moments binned in pT
        mask = False
        print("Saving for binning in pT")
    else:  # for systematic plots
        mask = True
        print("Saving for plotting the systematics")

    for mc_name in mc_names:

        ##### nominal
        if flags.nom:
            print("{}.h5".format(mc_name))    
            print("iteration: ", flags.niter)
            mc_info = MCInfo(mc_name,flags.N,flags.data_folder,config,use_fiducial_mask=mask)
            mfold = mc_info.LoadDataWeights(flags.niter,mode=flags.mode)
            
            print("nominal at iteration: ", flags.niter)
            model_strap = '/pscratch/sd/j/jing/H1PCT/weights_raw_300e6_median/{}_Pythia_nominal_iter{}_step2.h5'.format(
            base_name, flags.niter )
            weights =  mc_info.Reweight(mfold,model_name=model_strap)            
            with h5.File(os.path.join(flags.out,'nominal_iter{}.h5'.format(flags.niter)),'w') as fout:
                dset = fout.create_dataset('wgt', data=weights)
            del weights
            K.clear_session()
            del mc_info

        ##### systematics
        if flags.sys:
            print("{}.h5".format(mc_name))    
            for sys in opt.sys_sources:
                print(sys)
                mc_info = MCInfo(sys,flags.N,flags.data_folder,config,use_fiducial_mask=mask)
                mfold = mc_info.LoadDataWeights(flags.niter,mode=flags.mode)

                if sys == 'stat': continue
                model_strap = '/pscratch/sd/j/jing/H1PCT/weights_raw_300e6_median/{}_{}_iter{}_step2.h5'.format(
                base_name,sys,flags.niter)
                weights =  mc_info.Reweight(mfold,model_name=model_strap)            
                with h5.File(os.path.join(flags.out,'{}_iter{}.h5'.format(sys, flags.niter)),'w') as fout:
                    dset = fout.create_dataset('wgt', data=weights)
                del weights
                K.clear_session()
            del mc_info

        ##### straps
        # print("{}.h5".format(mc_name))    
        # mc_info = MCInfo(mc_name,flags.N,flags.data_folder,config)#,use_fiducial_mask=mask)
        # model_strap = '/pscratch/sd/j/jing/H1PCT/ensem_and_straps_weights/weights_strap/{}_{}_iter{}_step2_strapX.h5'.format(
        #     base_name,mc_name,flags.niter)
        # mfold = mc_info.LoadDataWeights(flags.niter,mode=flags.mode)
        # for nstrap in range(1,config['NBOOTSTRAP']+1):
        #     print(nstrap)
        #     weights =  mc_info.Reweight(mfold,model_name=model_strap.replace('X',str(nstrap)))            
        #     with h5.File(os.path.join(flags.out,'{}_{}.h5'.format(mc_name,nstrap)),'w') as fout:
        #         dset = fout.create_dataset('wgt', data=weights)
        #     del weights
        #     K.clear_session()
        # del mc_info

        ##### ensembling variance
        # mc_info = MCInfo(mc_name,flags.N,flags.data_folder,config)#,use_fiducial_mask=mask)
        # model_strap = '/pscratch/sd/j/jing/H1PCT/weights_trial/{}_{}_iter{}_step2_trialX.h5'.format(
        #     base_name,mc_name,flags.niter)
        # mfold = mc_info.LoadDataWeights(flags.niter,mode=flags.mode)
        # for ntrial in range(1,42):#config['NBOOTSTRAP']+1):
        #     print(ntrial)
        #     weights =  mc_info.Reweight(mfold,model_name=model_strap.replace('X',str(ntrial)))            
        #     with h5.File(os.path.join(flags.out,'{}_trial{}.h5'.format(mc_name,ntrial)),'w') as fout:
        #         dset = fout.create_dataset('wgt', data=weights)
        #     del weights
        #     K.clear_session()
        # del mc_info

        ##### closure
        # print("{}.h5".format(mc_name))    
        # mc_info = MCInfo(mc_name,flags.N,flags.data_folder,config,use_fiducial_mask=mask)
        # mfold = mc_info.LoadDataWeights(flags.niter,mode=flags.mode)
        # for i in [3,5]:
        #     print("iteration: ", i)
        #     model_strap = '/pscratch/sd/j/jing/H1PCT/weights/{}_Pythia_nominal_closure_iter{}_step2.h5'.format(
        #     base_name,i)
        #     weights =  mc_info.Reweight(mfold,model_name=model_strap)            
        #     with h5.File(os.path.join(flags.out,'closure_iter{}.h5'.format(i)),'w') as fout:
        #         dset = fout.create_dataset('wgt', data=weights)
        #     del weights
        #     K.clear_session()
        # del mc_info
