import os

#os.system("rm Unfolding/uncertaintiesBootstrap")
#os.system("make Unfolding/uncertaintiesBootstrap")
#os.system("rm -r Unfolding/mp/*")

in_tag  = "20220809"
out_tag = "20220809" # 20220615 correspond to 20220607 on BIG

the_config_file_path = '/pscratch/sd/j/jing/H1PCT/scripts_perlmutter/'

with open("jes_45.sh") as f:
    content = f.readlines()
lines = [x.strip() for x in content]
# with open("config_jes{}0.txt".format(i),"w") as task_list:

for i in range(0,45,5):
    print(i)
    with open(the_config_file_path+"jes_{}.sh".format(i),"w") as the_new_config:
        for j,l in enumerate(lines):
            # if(l[0]=='{'): continue
            # print(l.split())
            # print(j,l)
            # input()
            if j == 19: 
                l = "srun python Unfold.py --config config_jes{}.json --nevts 100e6".format(i)
                the_new_config.write(l+'\n')
            else:
                the_new_config.write(l+'\n')
