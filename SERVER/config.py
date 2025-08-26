class config:
    # Useful roots/dirs
    DATA_ROOT = '/workspace/dataset/cifar-10-batches-py/'

    # Data Spec
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 256
    
    # Training Spec
    LR = 1e-4

    # NTC Spec
    NUM_SCALES = 64
    SCALE_MIN = .11
    SCALE_MAX = 256.

    # JSCC Spec
    TRAIN_SNR = 10.
    NUM_SYMBOLS = 512
    
    # transformer Spec
    IMAGE_SHAPE = (32, 32, 3)
    PATCH_MERGE_SIZE = 2
    MLP_RATIO = 4
    CHANNEL_DIMENSION = 256
    NUM_BLOCKS_IN_LAYER = (2, 6)
    NUM_MHSA_HEADS = 8
    
    HOST = "***.***.***.***"
    PORT = 8080



    