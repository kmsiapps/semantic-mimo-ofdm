import argparse
import os
from tqdm import tqdm
import tensorflow as tf
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.model import SemViT_power
from utils.dataset import loadCifarDataset
from config import config
           
def load(clip):   
    model = SemViT_power(
        mean_coeff=     0,
        std_coeff=      0,
        papr_coeff=     0,
        clip_limit=     clip,
        snrdB=          10,
        num_symbols=    512,
        filters=        256,
        )
    model(tf.zeros([1, 32, 32, 3]))
    model.load_weights('./logs/Yoo_power-Mp2048/weights/epoch_96').expect_partial()
    return model

def papr_metric(model, test_ds):
    history = []
    for img, _ in tqdm(test_ds, leave=False):
        _, _, metric = model(img)
        history.append([metric['papr']])
    history = tf.concat(history, axis=1)
    history = tf.reduce_mean(history, axis=1)
    return history[0]

def limit_papr(test_ds, goal_papr, thres):
    # clip_end = 1000
    # model = load(clip_end)
    # papr_end = papr_metric(model, test_ds)
    # print(f"MAX PAPR : {papr_end}")
    # if papr_end < goal_papr:
    #     print("No PAPR Restrictions Required!")
    #     return
    
    clip_max = 2
    clip_min = 1
    
    while True:
        clip = (clip_max + clip_min) / 2
        model = load(clip)
        papr = papr_metric(model, test_ds)
        print(f"With clip = {clip:8.6f}, PAPR = {papr}")
        if abs(goal_papr - papr) < thres:
            print(f"Use Clip Limit as : {clip}")
            return
        if goal_papr < papr:
            clip_max = clip
        else:
            clip_min = clip
    
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('goal_papr', type=float)
    parser.add_argument('--threshold', type=float, default=0.0001)
    parser.add_argument('--mini', action='store_true', help='use mini dataset (with 200 images)')
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
    
    limit_papr(
        test_ds     = test_ds[0],
        goal_papr   = args.goal_papr,
        thres       = args.threshold
    )