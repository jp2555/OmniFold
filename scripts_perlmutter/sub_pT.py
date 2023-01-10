#! /usr/bin/env python
import os
# import commands
# import glob

# Put the list of datasets you want to run over here (remove the '/' from the end). 

for i in range(1,2):    

    # pTmin = 40+i*50

    cmd = 'python Plot_unfolded.py --plot_reco --q2_int %d ' %i

    print(cmd)
    os.system(cmd)
    print('\n')