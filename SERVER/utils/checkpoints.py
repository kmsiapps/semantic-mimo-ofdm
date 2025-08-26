import argparse
import os

def read_ckpt(exps):
    for exp in exps:
        with open(f'logs/{exp}/weights/checkpoint', 'r') as f:
            print(exp)
            print(f.readline().strip())

def delete_useless_ckpt(exps):
    for exp in exps:
        with open(f'logs/{exp}/weights/checkpoint', 'r') as f:
            print(exp)
            usefull_ckpt = f.readline().strip().split(' ')[1].strip('"')
            print(usefull_ckpt)
        lists = os.listdir(f'logs/{exp}/weights')
        
        for item in lists:
            if "epoch" in item and usefull_ckpt not in item:
                os.remove(f'logs/{exp}/weights/{item}')

def find_deletable_ckpt(*args):
    all_exps = os.listdir(f'logs')
    all_exps.remove('model_configs.json')
    for exp in all_exps:
        if os.path.isdir(f'logs/{exp}/weights'):
            num_ckpts = len(os.listdir(f'logs/{exp}/weights')) - 1
        else:
            num_ckpts = len(os.listdir(f'logs/{exp}')) - 4
        
        if num_ckpts > 2:
            print(exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('func_type', type=str)
    parser.add_argument('--exps', action='append', default=None, help='experiment names')
    args = parser.parse_args()
    
    exps = args.exps
    # exps = [    
    #     'Yoo_power-C0o9375', 
    #     'Yoo_power-C1', 
    #     'Yoo_power-C1o25', 
    #     'Yoo_power-C1o75', 
    # ]
    # exps = [    
    #     'Yoo_power-Pp16384', 
    #     'Yoo_power-Pp24576', 
    #     'Yoo_power-Pp32768', 
    #     'Yoo_power-Pp49152', 
    #     'Yoo_power-Pp65536', 
    #     'Yoo_power-Pp131072', 
    # ]
    # exps = [    
    #     'Yoo_power-Mp2048-Pp16384', 
    #     'Yoo_power-Mp2048-Pp24576', 
    #     'Yoo_power-Mp2048-Pp32768', 
    #     'Yoo_power-Mp2048-Pp49152', 
    #     'Yoo_power-Mp2048-Pp65536', 
    #     'Yoo_power-Mp2048-Pp131072', 
    # ]
    # exps = [    
    #     'Yoo_power-Mp2048-C0o75', 
    #     'Yoo_power-Mp2048-C0o875', 
    #     'Yoo_power-Mp2048-C0o9375', 
    #     'Yoo_power-Mp2048-C1', 
    #     'Yoo_power-Mp2048-C1o0625', 
    #     'Yoo_power-Mp2048-C1o125', 
    #     'Yoo_power-Mp2048-C1o25', 
    #     'Yoo_power-Mp2048-C1o5', 
    #     'Yoo_power-Mp2048-C1o75', 
    #     'Yoo_power-Mp2048-C2'
    # ]
    exps = [
    'USRP_quantize_0dB_mimic',
    'Yoo_base_train5',
    'Yoo_base_train25',
    'Yoo_base_train15',
    'USRP_base_fromclip',
    'Yoo_base_train20',
    'Yoo_power-Pp16384-train20',
    'Yoo_base_train0',
    'Yoo_base',
    'USRP_quantize_0dB',
    'USRP_base_meanpapr',
    'USRP_base']
   
    func = globals()[args.func_type]
    func(exps)