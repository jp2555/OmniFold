import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
#import uproot3 as uproot


dedicated_binning = {
    'gen_tf_c':np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    'gen_tf_f':np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    #'gen_tf_c':np.linspace(0,1,50),'gen_tf_f':np.linspace(0,1,50),
    #'gen_pt_c':np.array([170,190,220,800]),
    #'gen_pt_f':np.array([170,190,220,800]),
    'gen_trkpt_c':np.array([0,50,80,100,120,140,600]),
    'gen_trkpt_f':np.array([0,50,80,100,120,140,600]),
    #'gen_pt_c':np.linspace(100,500,20), 'gen_pt_f':np.linspace(100,500,20),
    #'gen_trkpt_c':np.linspace(0,500,25), 'gen_trkpt_f':np.linspace(0,500,25),
    # q/g fraction
    #'gen_pt_c':np.array([300, 600, 1100, 2500]),
    #'gen_pt_f':np.array([300, 600, 1100, 2500]),
    'gen_pt_c':np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]),
    'gen_pt_f':np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]),
    #'gen_pt_c':np.array([800, 900, 1000]),
    #'gen_pt_f':np.array([800, 900, 1000]),
    # log pT test
    #'gen_pt_c':np.linspace(5.25,6.5,21), 'gen_pt_f':np.linspace(5.25,6.5,21), # ~190 GeV
    #'gen_trkpt_c':np.linspace(2,7,21), 'gen_trkpt_f':np.linspace(2,7,21),
    'tf_c':np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), 
    'tf_f':np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),  
    #'tf_c':np.linspace(0,1,50),'tf_f':np.linspace(0,1,50),
    # q/g fraction
    #'pt_c':np.array([300, 600, 1100, 2500]),
    #'pt_f':np.array([300, 600, 1100, 2500]),
    'pt_c':np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]),
    'pt_f':np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000, 2500]),
    #'pt_c':np.array([800, 900, 1000]),
    #'pt_f':np.array([800, 900, 1000]),
    # log pT test
    #'pt_c':np.linspace(4,7,21),'pt_f':np.linspace(4,7,21),
    #'trkpt_c':np.linspace(2,7,21),'trkpt_f':np.linspace(2,7,21),
    #'pt_c':np.linspace(100,500,20), 'pt_f':np.linspace(100,500,20),
    'trkpt_c':np.linspace(0,500,25), 'trkpt_f':np.linspace(0,500,25),
}

fixed_yaxis = {
    'gen_tf_c':3, 'gen_tf_f':3,
    'tf_c':3, 'tf_f':3,
    'gen_pt_c':0.015, 'gen_pt_f':0.015,
    'gen_trkpt_c':0.015, 'gen_trkpt_f':0.015,
    # log pT test
    #'gen_pt_c':4, #'gen_pt_f':4,
    #'gen_trkpt_c':2, #'gen_trkpt_f':2,
    }

sys_sources = {
    'Pythia_UP': 'orange',
    'Pythia_DOWN': 'blue', # PURW
    #'Pythia_CORE': 'brown',#'Pythia_EDGE': 'yellowgreen', # TILE
    #'Sherpa_Lund': 'deepskyblue','Sherpa_AHADIC': 'blue', # Modeling
    #'Herwig_Dipole': 'red','Herwig_Angular': 'orchid', # Modeling
    #'Pythia_TRACK1': 'red',
    #'Pythia_TRACK2': 'orchid',
    #'Pythia_TRACK3': 'deepskyblue',
    #'Pythia_TRACK4': 'yellowgreen',   
    #'Hadronization':'orchid',
    #'Parton Shower': 'yellowgreen',
    #'stat':'tab:pink',
    #'ensem':'tab:pink',
    #'closure':'skyblue',
    #'Pythia_JES45': 'orange',
    #'Pythia_JES46': 'orangered',
    #'Pythia_JES47': 'deepskyblue',
    #'Pythia_JES48': 'yellowgreen',
    # 'Pythia_JES1': 'orangered','Pythia_JES2': 'deepskyblue','Pythia_JES3': 'yellowgreen','Pythia_JES4': 'orchid',
    # 'Pythia_JES5': 'orange','Pythia_JES6': 'orangered','Pythia_JES7': 'deepskyblue','Pythia_JES8': 'yellowgreen','Pythia_JES9': 'orchid',
    # 'Pythia_JES10': 'orange','Pythia_JES11': 'orangered','Pythia_JES12': 'deepskyblue','Pythia_JES13': 'yellowgreen','Pythia_JES14': 'orchid',
    # 'Pythia_JES15': 'orange','Pythia_JES16': 'orangered','Pythia_JES17': 'deepskyblue','Pythia_JES18': 'yellowgreen','Pythia_JES19': 'orchid',
    # 'Pythia_JES20': 'orange','Pythia_JES21': 'orangered','Pythia_JES22': 'deepskyblue','Pythia_JES23': 'yellowgreen','Pythia_JES24': 'orchid',
    # 'Pythia_JES25': 'orange','Pythia_JES26': 'orangered','Pythia_JES27': 'deepskyblue','Pythia_JES28': 'yellowgreen','Pythia_JES29': 'orchid',
    # 'Pythia_JES30': 'orange','Pythia_JES31': 'orangered','Pythia_JES32': 'deepskyblue','Pythia_JES33': 'yellowgreen','Pythia_JES34': 'orchid',
    # 'Pythia_JES35': 'orange','Pythia_JES36': 'orangered','Pythia_JES37': 'deepskyblue','Pythia_JES38': 'yellowgreen','Pythia_JES39': 'orchid',
    # 'Pythia_JES40': 'orange','Pythia_JES41': 'orangered','Pythia_JES42': 'deepskyblue','Pythia_JES43': 'yellowgreen','Pythia_JES44': 'orchid',
    #'Pythia_JES45': 'orange','Pythia_JES46': 'orangered','Pythia_JES47': 'deepskyblue',
    #'Pythia_JES48': 'yellowgreen',
    #'Pythia_JES49': 'orchid','Pythia_JES50': 'orange','Pythia_JES51': 'orangered','Pythia_JES52': 'deepskyblue',
    # 'Pythia_JES53': 'yellowgreen','Pythia_JES54': 'orchid','Pythia_JES55': 'orange','Pythia_JES56': 'orangered','Pythia_JES57': 'deepskyblue','Pythia_JES58': 'yellowgreen','Pythia_JES59': 'orchid',
    # 'Pythia_JES60': 'orange','Pythia_JES61': 'orangered','Pythia_JES62': 'deepskyblue','Pythia_JES63': 'yellowgreen','Pythia_JES64': 'orchid',
    # 'Pythia_JES65': 'orange','Pythia_JES66': 'orangered','Pythia_JES67': 'deepskyblue','Pythia_JES68': 'yellowgreen','Pythia_JES69': 'orchid',
    # 'Pythia_JES70': 'orange','Pythia_JES71': 'orangered','Pythia_JES72': 'deepskyblue','Pythia_JES73': 'yellowgreen','Pythia_JES74': 'orchid',
    # 'Pythia_JES75': 'orange','Pythia_JES76': 'orangered','Pythia_JES77': 'deepskyblue','Pythia_JES78': 'yellowgreen','Pythia_JES79': 'orchid',
    # 'Pythia_JES80': 'orange','Pythia_JES81': 'orangered','Pythia_JES82': 'deepskyblue','Pythia_JES83': 'yellowgreen','Pythia_JES84': 'orchid',
    # 'Pythia_JES85': 'orange','Pythia_JES86': 'orangered','Pythia_JES87': 'deepskyblue','Pythia_JES88': 'yellowgreen','Pythia_JES89': 'orchid',
    # 'Pythia_JES90': 'orange','Pythia_JES91': 'orangered','Pythia_JES92': 'deepskyblue','Pythia_JES93': 'yellowgreen','Pythia_JES94': 'orchid',
    # 'Pythia_JES95': 'orange','Pythia_JES96': 'orangered','Pythia_JES97': 'deepskyblue','Pythia_JES98': 'yellowgreen','Pythia_JES99': 'orchid',
    # 'Pythia_JES100': 'orange','Pythia_JES101': 'orangered','Pythia_JES102': 'deepskyblue','Pythia_JES103': 'yellowgreen','Pythia_JES104': 'orchid',
    # 'Pythia_JES105': 'orange','Pythia_JES106': 'orangered','Pythia_JES107': 'deepskyblue','Pythia_JES108': 'yellowgreen','Pythia_JES109': 'orchid',
     #'Pythia_JES110': 'orange',
     #'Pythia_JES111': 'orangered','Pythia_JES112': 'deepskyblue',
    }


# sys_translate = {
#     'sys_0':"Pythia_OFF",'sys_1':"Pythia_UP",'sys_2':"Pythia_DOWN",
#     'sys_3':"Pythia_CORE",'sys_4':"Pythia_EDGE",
#     'sys_5':"Sherpa_Lund", 'sys_6':"Sherpa_AHADIC",
#     'sys_7':'Pythia_TRACK1', 'sys_8':'Pythia_TRACK2', 'sys_9':'Pythia_TRACK3', 'sys_10':'Pythia_TRACK4', # TRACK
#     'sys_11':'Pythia_JES46', 'sys_12':'Pythia_JES47', 'sys_13':'Pythia_JES48', 'sys_14':'Pythia_JES49', # JES
#     # 'model': 'Model',
#     #'closure': 'Non-closure',
#     'stat':'Stat.',
# }


############## NAMES
name_translate = {
    'Sherpa': "Sherpa_Lund",
    'Herwig': "Herwig",
    'Herwig_Matchbox': "Herwig + Matchbox",
    'Pythia': 'Pythia',
    'Pythia_Vincia':'Pythia + Vincia',
    'Pythia_Dire':'Pythia + Dire',
}

reco_vars = {
    'tf_c':r'TF central$(p_T^{charged}/p_T^{all})$', 
    'tf_f':r'TF forward $(p_T^{charged}/p_T^{all})$', 
    'pt_c':r'pT central$(p_T^{all_c})$', 
    'pt_f':r'pT forward $(p_T^{all_f})$', 
    #'trkpt_c':r'track pT central$(p_T^{charged_c})$', 
    #'trkpt_f':r'track pT forward $(p_T^{charged_f})$', 
}

gen_vars = {
    'gen_tf_c':r'Gen TF central $(p_T^{charged}/p_T^{all})$', 
    'gen_tf_f':r'Gen TF forward $(p_T^{charged}/p_T^{all})$', 
    #'gen_pt_c':r'pT central$(p_T^{all_c})$', 
    #'gen_pt_f':r'pT forward $(p_T^{all_f})$', 
    #'gen_trkpt_c':r'track pT central$(p_T^{charged_c})$', 
    #'gen_trkpt_f':r'track pT forward $(p_T^{charged_f})$', 
}

var_truth_translate = {
    'gen_pt_c': "h1_tjet_pt_incl_central_nominal",
    'gen_pt_f': "h1_tjet_pt_incl_forward_nominal",
    'gen_trkpt_c': 'h1_tjet_trkpt_incl_central_nominal',
    'gen_trkpt_f': 'h1_tjet_trkpt_incl_forward_nominal',
    'gen_tf_c': 'h1_tjet_ChargedFrac_central_nominal',
    'gen_tf_f': 'h1_tjet_ChargedFrac_forward_nominal',
}


############## visualization
colors = {
    'LO':'b', 
    'NLO':'g',
    'NNLO':'r', 
    'Pythia_Vincia': '#9467bd',
    'Pythia_Dire': 'indigo',
    'Pythia':'blueviolet',
    'H7EG':'#8c564b',
    'Sherpa':'deepskyblue',
    'Herwig':'crimson',
    'Herwig_Matchbox':'crimson',
    'Cascade':'b',
    
    'PCT': 'g',    
    'standard':'blueviolet',
    'hybrid':'red',
}


markers = {
    'H7EG':'P',
    'Sherpa':'X',

    'Pythia': '^',
    'Pythia_Vincia': '<',
    'Pythia_Dire': '>',
    'Herwig':'D',
    'Herwig_Matchbox':'d',
    'PCT':'P',
    'standard':'o',
    'hybrid':'x',
}

#Shift in x-axis for visualization
xaxis_disp = {
    'Pythia': 0.0,
    'Sherpa': 0.0,
    'Pythia_Vincia': 0.3,
    'Pythia_Dire': 0.6,
    'Herwig':0.0,
    'Herwig_Matchbox':-0.6,
    'H7EG':-0.3,
}
    
def LoadFromROOT(file_name,var_name,q2_bin=0):
    with uproot.open(file_name) as f:
        if b'DIS_JetSubs;1' in f.keys():
            #Predictions from rivet
            hist = f[b'DIS_JetSubs;1']            
        else:
            hist = f
        if q2_bin ==0:
            var, bins =  hist[var_name].numpy()            
        else: #2D slice of histogram
            var =  hist[var_name+"2D"].numpy()[0][:,q2_bin-1]
            bins = hist[var_name+"2D"].numpy()[1][0][0]
            
        norm = 0
        for iv, val in enumerate(var):
            norm += val*abs(bins[iv+1]-bins[iv])
        return var
        
def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(npanels=2):
    fig = plt.figure(figsize=(9, 9))
    if npanels ==2:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    elif npanels ==3:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3,1,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    xposition = 0.8
    yposition=0.9
    # xposition = 0.83
    # yposition=1.03
    text = r'$\bf{ATLAS}$ Internal'
    WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             #fontweight='bold',
             transform = ax0.transAxes, fontsize=25)


    

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='None', ecolor='k')
