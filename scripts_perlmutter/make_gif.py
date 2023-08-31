from PIL import Image
from PIL import ImageDraw,ImageFont
import glob
import os
import argparse


parser = argparse.ArgumentParser()

#parser.add_argument('--folder', default='../plots_Rapgap_nominal_closure_hybrid', help='Folder containing figures')
parser.add_argument('--folder', default='/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/qg_full', help='Folder containing figures')
#parser.add_argument('--plot', default='gen_jet_tau15', help='Name of the distribution to plot')
parser.add_argument('--niter', type=int, default=3, help='Number of iterations to run over')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('--pt', type=int, default=12, help='Number of pT bins')
flags = parser.parse_args()


# Create the frames

base_folder = flags.folder
#to_gif = flags.plot
font = ImageFont.truetype("Helvetica-Bold.ttf", size=35)

plot_list = {
    'qg':r'qg',
    # 'gen_tf_c':r'$\mathrm{pTratio_{central}}$', 
    # 'gen_tf_f':r'$\mathrm{pTratio_{forward}}$', 
    # 'gen_pt_c':r'$p_\mathrm{T}_{\mathrm{central}}$',
    # 'gen_pt_f':r'$p_\mathrm{T}_{\mathrm{forward}}$', 
}

if flags.plot_reco == True:
    plot_list = {
    'tf_c':r'$\mathrm{pTratio_{central}}$', 
    'tf_f':r'$\mathrm{pTratio_{forward}}$', 
    'pt_c':r'$p_\mathrm{T}_{\mathrm{central}}$',
    'pt_f':r'$p_\mathrm{T}_{\mathrm{forward}}$', 
    }

for to_gif in plot_list:
    frames = []
    for i in range(0,flags.pt):
        new_frame = Image.open(os.path.join(base_folder,"{}_{}-1.png".format(to_gif,i)))
        # draw = ImageDraw.Draw(new_frame)
        # draw.text((120, 75), "Iteration {}".format(i),fill="black",font=font)
        frames.append(new_frame)
 
    # Save into a GIF file that loops forever
    frames[0].save(os.path.join(base_folder,'{}.gif'.format(to_gif)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)
