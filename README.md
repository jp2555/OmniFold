# Jet substructure with Omnifold

Repository to store the scripts used in the studies of jet observable unfolding using the H1 dataset. Main branch stores the strategy planned for the current iteration, while the ```perlmutter``` branch is used for development of Omnifold at scale. 

Current datasets are stored both at NERSC and at ML4HEP local machines. Those are already curated ```.h5``` files containing per particle and per jet information. 

* To set up the environment:
module load tensorflow/2.9.0 


* To run Omnifold use:

```bash
python Unfold.py  --data_folder FOLDER/CONTAINING/H5/FILES [--closure] --niter NITER
```

The flag ```closure``` runs the unfolding procedure without data files, but taking an MC (e.g. Sherpa) as the data representative. 

* To evaluate the save NN and save the weight:

python SaveWeights.py --data_folder FOLDER/CONTAINING/H5/FILES [--config CONFIG.json]

* Lastly to plot:

python Plot_unfolded.py --data_folder FOLDER/CONTAINING/H5/FILES [--config CONFIG.json] --niter NITER (--sys)

The outputs are the trained models for steps 1 and 2 for each omnifold iteration up to NITER, so one may plot a particular step to check if it is converging to the target in the closure test with the option --niter

* Or to plot in e.g. pT bins
python Plot_moment.py --sys

