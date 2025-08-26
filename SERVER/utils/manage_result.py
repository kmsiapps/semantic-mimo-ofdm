#%% 

def save_simulation(exp, val_name, SNRs, values, fname):
    words = f',{exp}\n'
    words = words + f'SNR,{val_name}\n'
    for SNR, value in zip(SNRs, values):
        words = words + f'{SNR:2} dB,{value}\n'
    with open(f'data/{fname}.csv', 'w') as f:
        f.write(words)

def draw_simulation(fname):
    with open(f'../data/{fname}.csv', 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    exp = lines[0].strip()[1]
    val_name = lines[1].strip()
    for line in lines[2:]:
        line = line.strip().split(',')
        SNRs.append(float(line[0][:-3]))
        values.append(float(line[1]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, marker='o', label=f'{exp} {val_name}')
    
    plt.title(f'{val_name} of Various Models')
    plt.xlabel('SNR')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.show()
    
def read_file(fname):
    with open(f'../data/{fname}.csv', 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    exp = lines[0].strip()[1]
    val_name = lines[1].strip()
    for line in lines[2:]:
        line = line.strip().split(',')
        SNRs.append(float(line[0][:-3]))
        values.append(float(line[1]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    
    return SNRs, values

#%% 
if __name__=='__main__':
    draw_simulation('psnr_sample_img')

# %%
if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    USRP_file = f'../data/DNN_USRP_TEST.csv'
    SIM_file = f'../data/psnr_sample_esnr.csv'

    with open(SIM_file, 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    exp = lines[0].strip()[1]
    val_name = lines[1].strip()
    for line in lines[2:]:
        line = line.strip().split(',')
        SNRs.append(float(line[0][:-3]))
        values.append(float(line[1]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, marker='o', label=f'simulation data')

    with open(USRP_file, 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    for line in lines[1:]:
        line = line.strip().split(',')
        SNRs.append(float(line[1])-3)
        values.append(float(line[3]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, '.', label=f'usrp data')

    plt.title(f'PSNR score Simulation vs USRP')
    plt.xlabel('SNR')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.show()
    
    
# %%

if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    USRP_file = f'../data/240129USRP.txt'
    SIM_file = f'../data/psnr_sample_esnr.csv'

    compensation = 1.5

    with open(USRP_file, 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    for line in lines[1:]:
        line = line.strip().split(' ')
        SNRs.append(float(line[2])-compensation)
        values.append(float(line[8]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, '.', label=f'usrp data')

    with open(SIM_file, 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    exp = lines[0].strip()[1]
    val_name = lines[1].strip()
    for line in lines[2:]:
        line = line.strip().split(',')
        SNRs.append(float(line[0][:-3]))
        values.append(float(line[1]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, marker='', label=f'simulation data')

    plt.title(f'PSNR score Simulation vs USRP')
    plt.xlabel('Effective SNR')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.show()

# %%

if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    USRP_file = f'../data/240207USRP.txt'
    SIM_file = f'../data/psnr_sample_adap.csv'

    compensation = 1.5

    with open(USRP_file, 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    for line in lines[1:]:
        line = line.strip().split(' ')
        SNRs.append(float(line[2])-compensation)
        values.append(float(line[8]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, '.', label=f'usrp data')

    with open(SIM_file, 'r') as f:
        lines = f.readlines()
    SNRs = []
    values = []
    exp = lines[0].strip()[1]
    val_name = lines[1].strip()
    for line in lines[2:]:
        line = line.strip().split(',')
        SNRs.append(float(line[0][:-3]))
        values.append(float(line[1]))
    SNRs, values = zip(*sorted(zip(SNRs, values)))
    plt.plot(SNRs, values, marker='o', label=f'simulation data')

    # plt.title(f'PSNR score Simulation vs USRP')
    plt.xlabel('Effective SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('../images/final/USRP_vs_SIM', dpi=300)
    
# %%
