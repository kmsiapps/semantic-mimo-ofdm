import argparse
import os
import copy
import math
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.load import *
from utils.dataset import loadCifarDataset, loadCifarValidImage
from config import config

### Figure Plotting Functions ###

# ------------------------------ Fig 3
# python figure.py error_4x4_MIMO_shuffling
def error_4x4_MIMO_shuffling(**kwargs):
    '''
    Error with / without interleaving in 4x4 MIMO channel.
    '''
    # Data preprocessing
    noise_result = np.load('data/result_4x4_noise.npz')
    keys = ['sc_wise_noise', 'shuf_sc_wise_noise']
    subcarrier_idx = np.concatenate((np.arange(28, 64), np.arange(65, 101)))
    
    tx_maps = noise_result['resource_maps']
    rx_maps = noise_result['resource_maps_rcv_zf']
    df_maps = tx_maps - rx_maps
    err_maps = np.abs(df_maps)**2
    snr_maps = err_maps / np.abs(rx_maps+1e-8)**2
    
    map_shape = err_maps.shape
    shuf_err_maps = np.reshape(copy.deepcopy(err_maps), (-1))
    np.random.shuffle(shuf_err_maps)
    shuf_err_maps = np.reshape(shuf_err_maps, map_shape)
    
    mean_err = np.mean(err_maps, axis=(1, 2, 3))
    sym_pow = np.mean(np.abs(tx_maps)**2, axis=(1, 2, 3, 4))
    for i in range(4):
        esnr_db = 10 * np.log10(sym_pow[i] / np.mean(mean_err[i]))
        mean_snr_db = 10 * np.log10(mean_err[i]) / sym_pow[i]
        plt.plot(subcarrier_idx, mean_snr_db, label=f'Ant {i}, w/o shuffling, Rx SNR : {esnr_db:5.2f}dB')  
        
    mean_snr = np.mean(shuf_err_maps, axis=(1, 2, 3))
    sym_pow = np.mean(np.abs(tx_maps)**2)
    esnr_db = 10 * np.log10(np.mean(sym_pow / np.mean(mean_snr)))
    for i in range(4):
        mean_snr_db = 10 * np.log10(mean_snr[i])
        if i == 0:
            plt.plot(subcarrier_idx, mean_snr_db, color='gray', label=' '*7+f'w/  shuffling, Rx SNR : {esnr_db:5.2f}dB') 
        else:
            plt.plot(subcarrier_idx, mean_snr_db, color='gray') 
        
    plt.ylim([-26, 9])
    plt.ylabel('Signal-relative Noise Power [dB]')
    plt.xlabel('Subcarrier Index')
    plt.grid()
    plt.legend(prop={'family':'monospace'}, loc='upper right')
    plt.savefig('figures/3_mimo_noise_power.png', dpi=300)

    
# ------------------------------ Fig 4
# python figure.py ofdm_papr_constellations
def ofdm_papr_constellations(test_ds, **kwargs):
    '''
    Figure plot for [Fig. 4.(a)]
    A scatter of semantic constellations in time domain, of single image. 
    SNR=20dB trained model is used.
    P-value of PAPR reduction model is 1/4096, 1/32768 for strong, weak models, respectively.
    Also PAPR and PSNR values are shown in the plot legend, which are averaged from CIFAR-10 validation dataset
    '''
    
    FFT_size = 72
    num_symbols = 512
    scatter_image = loadCifarValidImage(config.DATA_ROOT, 3278)
    experiments = ['Yoo_base_train20', 'SemViT_20_Pp32768', 'SemViT_20_Pp4096']
    colors = ['mediumorchid', 'c', '#FF7900']
    face = ['none', None, 'none']
    markers = ['o', 'x', 'D']
    captions = ['Baseline', 'Weak', 'Strong']
    LW = [1, 0.7, 0.8]
    
    def getSymbols(model, img):
        y = model.enc0(img)
        y = model.enc1(y)
        y = model.enc21(y)
        y = model.enc22(y)
        y = model.enc23(y)
        y = model.enc_proj(y)
        y = tf.reshape(y, (-1, 2))
        y_f = tf.complex(y[:,0], y[:,1])
        y_f = tf.reshape(y_f, (-1, num_symbols)) #
        y_f = y_f[:,:num_symbols//FFT_size*FFT_size] #
        y_f = tf.reshape(y_f, (-1, num_symbols//FFT_size, FFT_size))
        y_t = tf.signal.ifft(y_f)
        return y_t
        
    fig_scatter = plt.figure(figsize=(5, 5))
    ax_scatter = fig_scatter.add_subplot(1, 1, 1)
    for i, exp in enumerate(experiments):
        model, _ = loadModel(exp)
        # Get mean PAPR and PSNR
        PAPR = []
        PSNR = []
        for img, _ in tqdm(test_ds, leave=False):
            y_t = getSymbols(model, img)
            y_pow = tf.abs(y_t)**2
            papr = tf.reduce_max(y_pow, axis=2) / tf.reduce_mean(y_pow, axis=2)
            PAPR.append(tf.reshape(papr, (-1)))
            _, _, metric = model(img, training=True)
            PSNR.append(metric['psnr'])
        PAPR = tf.reduce_mean(tf.concat(PAPR, 0))
        PAPRdB = 10 * tf.math.log(PAPR) / tf.math.log(10.)
        PSNR = tf.reduce_mean(tf.concat(PSNR, 0))
        # Get constellation
        y_t = getSymbols(model, scatter_image)
        y_t = tf.reshape(y_t, (-1))
        y_t /= tf.complex(tf.reduce_mean(tf.abs(y_t)**2)**0.5, tf.zeros(1, dtype=tf.float32))
        # print(f"Exp {exp:15}'s PAPR : {float(img_papr):5.2f} dB / PSNR : {float(img_psnr):5.2f} dB")
        ax_scatter.scatter(tf.math.real(y_t), tf.math.imag(y_t), color=colors[i], facecolors=face[i], marker=markers[i], linewidth=LW[i], \
            s=7, label=f"{captions[i]:8} (PAPR:{float(PAPRdB):4.2f}dB / PSNR:{float(PSNR):.2f}dB)")
        
    ranger = 6
    ax_scatter.legend(prop={'family':'monospace'})
    ax_scatter.set_xlim([-ranger, ranger])
    ax_scatter.set_ylim([-ranger, ranger])
    ax_scatter.set_xticks(np.arange(-ranger, ranger+0.01, ranger/2))
    ax_scatter.set_yticks(np.arange(-ranger, ranger+0.01, ranger/2))
    ax_scatter.grid()
    ax_scatter.set_xlabel('In-phase')
    ax_scatter.set_ylabel('Quadrature-phase')
    fig_scatter.tight_layout()
    fig_scatter.savefig(f'figures/4_PAPR_constellations.png', dpi=300)
    


# ------------------------------ Fig 6.a
# python figure.py result_RxSNR2PSNR
def result_RxSNR2PSNR(**kwargs):
    '''
    Figure plot for [Fig. 6.(a)]
    Plotting Rx SNR to PSNR performance.
    Data are stored in result_semantic.npz and result_bpg.csv file.
    Shows the power of interleaving.
    '''
    
    # Data preprocessing
    semantic = np.load('data/result_semantic.npz')
    keys = list(semantic.keys())
    results = []
    for key in keys:
        rlt = {}
        mean_result = np.mean(np.reshape(semantic[key], (-1, 10, 2)), axis=1)
        rlt['ESNR'] = mean_result[..., 0]
        rlt['PSNR'] = mean_result[..., 1]
        results.append(rlt)
    with open('data/result_bpg.csv', 'r') as f:
        lines = f.readlines()
    rlt = {}
    rlt['ESNR'] = []
    rlt['PSNR'] = []
    for line in lines[1:]:
        val = line.strip().split(',')
        rlt['ESNR'].append(float(val[0]))
        rlt['PSNR'].append(float(val[1]))
    results.append(rlt) 
    
    # Plotting
    plt.figure(figsize=(5.5, 5))
    captions = ['Semantic, w/ shuffling', 'Semantic, w/o shuffling', 'Semantic, AWGN', \
        'BPG + LDPC + QAM, w/ shuffling', 'BPG + LDPC + QAM, w/o shuffling']
    colors = iter(['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:gray'])
    for result in results:
        plt.plot(result['ESNR'], result['PSNR'], 'D-', color=next(colors))
    plt.plot(rlt['ESNR'], rlt['PSNR'], 'D:', linewidth=3.0, color=next(colors))
    plt.legend(captions)
    plt.xlabel('Rx SNR [dB]')
    plt.ylabel('PSNR [dB]')
    plt.grid()
    plt.savefig('figures/6a_RxSNR_PSNR.png', dpi=300)

    
# ------------------------------ Fig 6.b
# python figure.py result_PAPR
def result_PAPR(**kwargs):
    '''
    Figure plot for [Fig. 6.(b)]
    Plotting PA input power to Rx SNR and PSNR performance.
    Double Axes plotting is used. 
    Data are stored in result_RxSNR_PSNR.npz file.
    Shows the performance of PAPR reduction model.
    '''
    
    result = np.load('data/result_RxSNR_PSNR.npz')
    keys = list(result.keys())
    s = 3

    colors = ['mediumorchid', 'c', '#FF7900']
    captions = ['Baseline', 'Weak PAPR Reduction', 'Strong PAPR Reduction']
    markers = ['o', 'x', 'D']
    
    fig_PSNR = plt.figure(figsize=(5.5, 5))
    ax_PSNR = fig_PSNR.add_subplot(1, 1, 1)
    ax_ESNR = ax_PSNR.twinx()

    Tx_dB_ref = -20 * np.log10(1 / 3) + 8 # + 28.6
    for idx, key in enumerate(keys):
        mean_result = np.mean(np.reshape(result[key], (-1, 10, 3)), axis=1)
        Tx_dB = 10 * np.log10(1 / mean_result[..., 0] ** 2) + Tx_dB_ref
        ax_PSNR.plot(Tx_dB[:-2], mean_result[2:, 2], f'{markers[idx]}-', markersize=s, alpha=1.0, color=colors[idx])
        ax_ESNR.plot(Tx_dB[:-2], mean_result[2:, 1], f'{markers[idx]}:', markersize=s, alpha=1.0, color=colors[idx])

    ax_PSNR.set_xlabel('PA Input, Baseband Power [dBm]')
    ax_PSNR.set_ylabel('PSNR [dB]')
    ax_ESNR.set_ylabel('Rx SNR [dB]')
    ax_PSNR.set_ylim([24, 37])
    ax_ESNR.set_ylim([15, 28])
    ax_ESNR.set_yticks(np.arange(15, 27.5, 2))
    ax_PSNR.grid()
    ax_PSNR.legend(captions, loc='lower center')
    fig_PSNR.tight_layout()
    fig_PSNR.savefig('figures/6b_PAPR_reduction.png', dpi=300)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('valid_type', type=str)
    parser.add_argument('--mini', action='store_true', help='use mini dataset (with 200 images)')
    parser.add_argument('--plot', action='store_true', help='use plotting (for few functions)')
    parser.add_argument('--test_bs', type=int, default=config.TEST_BATCH_SIZE)
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT)
    args = parser.parse_args()
    
    test_ds = loadCifarDataset(
        None,
        args.test_bs,
        args.data_root,
        type='test',
        mini=args.mini
    )
    
    valid_function = globals()[args.valid_type]
    
    valid_function(
        test_ds     = test_ds[0],
        plot        = args.plot
    )