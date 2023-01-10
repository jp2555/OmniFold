import os

#os.system("rm Unfolding/uncertaintiesBootstrap")
#os.system("make Unfolding/uncertaintiesBootstrap")
#os.system("rm -r Unfolding/mp/*")

in_tag  = "20220809"
out_tag = "20220809" # 20220615 correspond to 20220607 on BIG

the_config_file_path = '/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/'

with open("config_jes.json") as f:
    content = f.readlines()
lines = [x.strip() for x in content]

# with open("config_jes{}0.txt".format(i),"w") as task_list:

for i in range(0,12):
    print(i)
    with open(the_config_file_path+"config_jes{}0.json".format(i),"w") as the_new_config:
        for j,l in enumerate(lines):
            # if(l[0]=='{'): continue
            # print(l.split())
            if j == 1: 
                l = "'MC_NAMES':['Pythia_JES{}0','Pythia_JES{}1','Pythia_JES{}2','Pythia_JES{}3','Pythia_JES{}4'],".format(i,i,i,i,i)
                the_new_config.write(l+'\n')
            else:
                the_new_config.write(l+'\n')
