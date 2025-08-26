import argparse
import os
import tensorflow as tf
from models.model import *
from utils.dataset import loadCifarDataset
from config import config

from utils.image import loadSampleImage, imBatchtoImage

def main(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load CIFAR-10 dataset
    train_ds, test_ds = loadCifarDataset(
        args.train_bs,
        args.test_bs,
        args.data_root,
        type='all',
        mini=args.mini
    )

    EXPERIMENT_NAME = args.experiment_name
    print(f'Running {EXPERIMENT_NAME}')
 
    model = globals()[args.model](    
        # Limitation Learning
        mean_coeff=     args.mean_coeff,
        std_coeff=      args.std_coeff,
        papr_coeff=     args.papr_coeff, 
        clip_limit=     args.clip_limit, 
        # Channel model 
        amp_fluc=       args.amp_fluc,  
        phase_fluc=     args.phase_fluc, 
        shape_param=    args.shape_param,  
        # Basic SemViT params
        snrdB=          args.train_snrdB,
        num_symbols=    args.num_symbols,
        filters=        args.channel_dimension,
    )      
    
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=args.learning_rate
        ),
        metrics=[
            psnr
        ]
    )

    model(tf.zeros([1, 32, 32, 3]))
    # model.build(input_shape=(None, 32, 32, 3))
    if args.freeze_encoder:
        model.encoder.trainable = False
    model.summary()

    if args.ckpt is not None:
        model.load_weights(args.ckpt)
    # model.norm = tf.Variable(args.normalizer * tf.ones((1,)), trainable=False, name='normalizer')

    save_ckpt = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./logs/{EXPERIMENT_NAME}/weights/epoch_" + "{epoch}",
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True,
            options=tf.train.CheckpointOptions(
                experimental_io_device=None, experimental_enable_async_checkpoint=True
            )
        )
    ]

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{EXPERIMENT_NAME}')
    history = model.fit(
        train_ds,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        callbacks=[tensorboard, save_ckpt],
        validation_data=test_ds,
    )
    
    model.save_weights(f"logs/{EXPERIMENT_NAME}/final")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='experiment name (used for ckpt & logs)')
    parser.add_argument('--model', type=str, help='model (JSCC or JSCC_norm)')
    parser.add_argument('--learning_rate', type=float, default=config.LR, help='learning rate')
    
    parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file path (optional)')
    parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
    parser.add_argument('--epochs', type=int, default=200, help='total epochs')
    
    parser.add_argument('--train_bs', type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument('--test_bs', type=int, default=config.TEST_BATCH_SIZE)
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT)
    parser.add_argument('--mini', action='store_true', help='use mini dataset (with 100 images each)')
    
    parser.add_argument('--channel_type', type=str, default='AWGN', help='channel type ("AWGN" or "Rician")')
    parser.add_argument('--train_snrdB', type=float, default=config.TRAIN_SNR, help='train snr (in dB)')
    parser.add_argument('--shape_param', type=float, default=0.1)
    
    parser.add_argument('--Rx_precision', type=int, default=16)
    parser.add_argument('--clipping_type', type=str, default='simple')
    parser.add_argument('--attenuation', type=float, default=1.)
    parser.add_argument('--normalizer', type=float)
    parser.add_argument('--freeze_encoder', action='store_true')
    
    parser.add_argument('--goal_power', type=float, default=0.5)
    parser.add_argument('--mean_coeff', type=float, default=0) 
    parser.add_argument('--papr_coeff', type=float, default=0)
    parser.add_argument('--clip_coeff', type=float, default=0)
    parser.add_argument('--std_coeff', type=float, default=0)  
    parser.add_argument('--clip_limit', type=float)
    parser.add_argument('--amp_fluc', type=float, default=0)
    parser.add_argument('--phase_fluc', type=float, default=0)
    
    parser.add_argument('--channel_dimension', type=int, default=config.CHANNEL_DIMENSION)
    parser.add_argument('--num_symbols', type=int, default=config.NUM_SYMBOLS)
    # parser.add_argument('--image_shape', type=tuple, default=config.IMAGE_SHAPE)
    # parser.add_argument('--patch_size', type=int, default=config.PATCH_MERGE_SIZE)
    # parser.add_argument('--num_blocks_in_layer', type=tuple, default=config.NUM_BLOCKS_IN_LAYER)
    # parser.add_argument('--num_mhsa_heads', type=int, default=config.NUM_MHSA_HEADS)
    # parser.add_argument('--mlp_ratio', type=int, default=config.MLP_RATIO)

    args = parser.parse_args()
    main(args)