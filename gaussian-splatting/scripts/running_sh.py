import os
from time import sleep

# python /path/to/your/gaussian-splatting/scripts/running_sh.py > "/path/to/your/gaussian-splatting/scripts/running.log" 2>&1 &
# nohup python /path/to/your/gaussian-splatting/scripts/running_sh.py > "/path/to/your/gaussian-splatting/scripts/running.log" 2>&1 &

sh_root = "/path/to/your/gaussian-splatting/data/mvimgnet/sh/"
# sh_root = "/path/to/your/gaussian-splatting/data/ABO/sh/"

running_list = range(1,1+1)
# running_list = [68]

for i in running_list:

    sh_file = sh_root+"run_"+str(i)+".sh"

    os.system("sh "+sh_file)
    print("completing file run_"+str(i)+".sh\n", flush=True)
    sleep(20)

print("over with i : %d(complete)"%(i))