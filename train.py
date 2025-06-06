import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../', "lib")))
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import sys
import argparse
import socket
from contextlib import closing

from torch.utils.tensorboard import summary
'''
ps -xH | grep multiprocessing | awk '{print $1}' | xargs kill -9
ps -xH | grep tensorboard | awk '{print $1}' | xargs kill -9

'''

# import warnings
# warnings.filters("ignore")

def args_to_str(args):
    argv = [args.config]
    if args.work_dir is not None:
        argv += ['--work-dir', args.work_dir]
    if args.resume_from is not None:
        argv += ['--resume-from', args.resume_from]
    if args.seed is not None:
        argv += ['--seed', str(args.seed)]
    return argv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--gpu-ids',type=int,nargs='+',help='ids of gpus to use')
    parser.add_argument('--seed', type=int, help='random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    
    if len(gpu_ids) == 1:
        import lib.train
        os.environ['training_script'] = './lib/train.py'
        if 'LOCAL_RANK' not in os.environ: 
            os.environ['LOCAL_RANK'] = "0"
            os.environ['RANK'] = "0"
        sys.argv = [''] + args_to_str(args)
        lib.train.main()
    else:
        from torch.distributed import launch
        for port in range(29500, 65536):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                res = sock.connect_ex(('localhost', port))
                if res != 0:
                    break
        os.environ['training_script'] = './lib/train.py'
        sys.argv = ['',
                    '--nproc_per_node={}'.format(len(gpu_ids)),
                    '--master_port={}'.format(port),
                    './lib/train.py'
                    ] + args_to_str(args) + ['--launcher', 'pytorch', "--world_size", str(len(gpu_ids))]
        launch.main()
    

if __name__ == "__main__":
    main()
    