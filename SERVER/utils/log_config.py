import json

model_parameters = {
    'indiv_norm': {
        'model_class':'JSCC',        
        'model_params':{
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_class':'JSCC_norm_val',   
        'valid_params':{
            'lamda'         : 1,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':195,
        'caption':'Baseline'
    },
    'learn_norm_L1': {
        'train_class':'JSCC_norm',        
        'train_params':{
            'lamda'         : 1,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_class':'JSCC_norm_val',   
        'valid_params':{
            'lamda'         : 1,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':200,
        'caption':'Norm Learning (L = 1)'
    },
    'learn_norm_L16': {
        'model_class':'JSCC_norm',        
        'model_params':{
            'lamda'         : 16,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_class':'JSCC_norm_val',   
        'valid_params':{
            'lamda'         : 16,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':198,
        'caption':'Norm Learning (L = 16)'
    },
    'learn_norm_L64': {
        'train_class':'JSCC_norm',        
        'train_params':{
            'lamda'         : 64,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_class':'JSCC_norm_val',   
        'valid_params':{
            'lamda'         : 64,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':197,
        'caption':'Norm Learning (L = 64)'
    },
    'learn_norm_L256': {
        'train_class':'JSCC_norm',        
        'train_params':{
            'lamda'         : 256,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_class':'JSCC_norm_val',   
        'valid_params':{
            'lamda'         : 256,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':198,
        'caption':'Norm Learning (L = 256)'
    },
    
    'learn_power_Mp16_Sp0': {
        'model_class':'JSCC_power',        
        'model_params':{
            'mean_coeff'    : 0.0625,
            'std_coeff'     : 0,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':197,
        'caption':'Power Learning (M=1/16, S=0)'
    },
    'learn_power_Mp16_Sp64': {
        'model_class':'JSCC_power',        
        'model_params':{
            'mean_coeff'    : 0.0625,
            'std_coeff'     : 0.015625,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':191,
        'caption':'Power Learning (M=1/16, S=1/64)'
    },
    'learn_power_Mp16_Sp256': {
        'model_class':'JSCC_power',        
        'model_params':{
            'mean_coeff'    : 0.0625,
            'std_coeff'     : 0.00390625,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':200,
        'caption':'Power Learning (M=1/16, S=1/256)'
    },
    'learn_power_Mp16_Sp1024': {
        'model_class':'JSCC_power',        
        'model_params':{
            'mean_coeff'    : 0.0625,
            'std_coeff'     : 0.0009765625,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':200,
        'caption':'Power Learning (M=1/16, S=1/1024)'
    },
    
    #####
        
    'JSCC-usrp_M0_S0_Cp16-b4-l2': {
        'model_class':'JSCC_usrp',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 2,
            'clip_coeff'    : 0.0625,
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':199,
        'caption':'USRP (M=0, S=0, C=1/16, pre=4bit, limit=2)'
    },    
    'JSCC-usrp_Mp64_S16384_Cp16-b4-l2': {
        'model_class':'JSCC_usrp',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 2,
            'clip_coeff'    : 0.0625,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':195,
        'caption':'USRP (M=1/64, S=1/16384, C=1/16, pre=4bit, limit=2)'
    },    
    'JSCC-usrp-sim_M0_S0_Cp16-b4-l2': {
        'model_class':'JSCC_usrp_simple',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 2,
            'clip_coeff'    : 0.0625,
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':200,
        'caption':'USRP-simple (M=0, S=0, C=1/16, pre=4bit, limit=2)'
    },    
    'JSCC-usrp-sim_Mp64_S16384_Cp16-b4-l2': {
        'model_class':'JSCC_usrp_simple',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 2,
            'clip_coeff'    : 0.0625,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snr_db'        : 10,
            'num_symbols'   : 512,
            
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':198,
        'caption':'USRP-simple (M=1/64, S=1/16384, C=1/16, pre=4bit, limit=2)'
    },    
    
    #####
    
    'Yoo-base': {
        'model_class':'SemViT',        
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':195,
        'caption':'SemViT base'
    }, 
    'Yoo-usrp_Mp64_S0_Cb4-l2': {
        'model_class':'SemViT_usrp',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 2,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':190,
        'caption':'USRP (M=1/64, S=0, pre=4bit, limit=2)'
    },
    'Yoo-usrp_Mp64_Sp16384_Cb4-l2': {
        'model_class':'SemViT_usrp', 
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 2,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':188,
        'caption':'USRP (M=1/64, S=1/16384, pre=4bit, limit=2)'
    },
    'Yoo-usrp_Mp64_Sp16384_Cb4-l3': {
        'model_class':'SemViT_usrp',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 3,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':198,
        'caption':'USRP (M=1/64, S=1/16384, pre=4bit, limit=3)'
    },
    'Yoo-usrp_Mp64_Sp16384_Cb4-l4': {
        'model_class':'SemViT_usrp',        
        'model_params':{
            'precision'     : 4,
            'clip_limit'    : 4,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':200,
        'caption':'USRP (M=1/64, S=1/16384, pre=4bit, limit=4)'
    },
    
    ##### Real model
    
    'real_channel-M0-S0-Cl8-Fao15_ho05': {
        'model_class':'SemViT_real',        
        'model_params':{
            'amp_fluc'      : 0.15,
            'phase_fluc'    : 0.05,
            'clip_limit'    : 8,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':8,
        'caption':'Real Channel (M=0, S=0, Limit=8, AF=0.15, AP=0.05)'
    },
    'rice_channel_indiv-M0-S0-Cl8-Fo13': {
        'model_class':'SemViT_rician',        
        'model_params':{
            'shape_param'   : 0.13,
            'clip_limit'    : 8,
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':10,
        'caption':'Rician Channel (M=0, S=0, Limit=8, K=0.13)'
    },
  
    #####
        
    'Yoo_base': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':700,
        'caption':'SemViT base'
    },  
    'Yoo_power-Mp4': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.25,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':95,
        'caption':'SemViT (M = 1/4)'
    },  
    'Yoo_power-Mp8': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.125,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':71,
        'caption':'SemViT (M = 1/8)'
    }, 
    'Yoo_power-Mp16': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0625,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':99,
        'caption':'SemViT (M = 1/16)'
    },  
    'Yoo_power-Mp32': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.03125,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':76,
        'caption':'SemViT (M = 1/32)'
    },  
    'Yoo_power-Mp64': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.015625,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':257,
        'caption':'SemViT (M = 1/64)'
    },  
    'Yoo_power-Mp256': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.00390625,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':95,
        'caption':'SemViT (M = 1/256)'
    },  
    'Yoo_power-Mp2048': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.00048828125,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':96,
        'caption':'SemViT (M = 1/2048)'
    }, 
    
    #####
    
    'Yoo_power-Sp16384': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':100,
        'caption':'SemViT (S = 1/16384)'
    },      
    'Yoo_power-Sp4096': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0.000244140625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':70,
        'caption':'SemViT (S = 1/4096)'
    },      
    'Yoo_power-Sp1024': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0.0009765625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':69,
        'caption':'SemViT (S = 1/1024)'
    },      
    'Yoo_power-Sp256': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0.00390625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':99,
        'caption':'SemViT (S = 1/256)'
    },      
    'Yoo_power-Sp64': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0.015625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':90,
        'caption':'SemViT (S = 1/64)'
    },      
    'Yoo_power-Sp16': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0.0625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':63,
        'caption':'SemViT (S = 1/16)'
    },
    
    ### PAPR Model
    'Yoo_power-Mp2048-Pp16': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':195,
        'caption':'SemViT (P = 1/16)'
    },
    'Yoo_power-Mp2048-Pp32': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0312,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':167,
        'caption':'SemViT (P = 1/32)'
    },
    'Yoo_power-Mp2048-Pp64': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.015625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':184,
        'caption':'SemViT (P = 1/64)'
    },
    'Yoo_power-Mp2048-Pp128': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0078125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':190,
        'caption':'SemViT (P = 1/128)'
    },
    'Yoo_power-Mp2048-Pp256': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00390625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':198,
        'caption':'SemViT (P = 1/256)'
    },
    'Yoo_power-Mp2048-Pp512': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.001953125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':169,
        'caption':'SemViT (P = 1/512)'
    },
    'Yoo_power-Mp2048-Pp1024': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0009765625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':197,
        'caption':'SemViT (P = 1/1024)'
    },
    'Yoo_power-Mp2048-Pp2048': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00048828125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':200,
        'caption':'SemViT (P = 1/2048)'
    },
    'Yoo_power-Mp2048-Pp4096': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.000244140625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':189,
        'caption':'SemViT (P = 1/4096)'
    },
    'Yoo_power-Mp2048-Pp8192': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0001220703125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':194,
        'caption':'SemViT (P = 1/8192)'
    },
    'Yoo_power-Mp2048-Pp16384': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':195,
        'caption':'SemViT (P = 1/16384)'
    },
    'Yoo_power-Mp2048-Pp24576': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00004069010416666667,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':197,
        'caption':'SemViT (P = 1/24576)'
    },
    'Yoo_power-Mp2048-Pp32768': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.000030517578125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':174,
        'caption':'SemViT (P = 1/32768)'
    },
    'Yoo_power-Mp2048-Pp49152': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00002034505208333333,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':187,
        'caption':'SemViT (P = 1/49152)'
    },
    'Yoo_power-Mp2048-Pp65536': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0000152587890625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':199,
        'caption':'SemViT (P = 1/65536)'
    },
    'Yoo_power-Mp2048-Pp131072': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00000762939453125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':172,
        'caption':'SemViT (P = 1/131072)'
    },    
    
    'Yoo_power-Pp16384': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00006103515625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':196,
        'caption':'SemViT with PAPR Loss (λ = 1/16384)'
    },
    'Yoo_power-Pp24576': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00004069010416666667,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':176,
        'caption':'SemViT (P = 1/24576)'
    },
    'Yoo_power-Pp32768': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.000030517578125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':192,
        'caption':'SemViT with PAPR Loss (λ = 1/32768)'
    },
    'Yoo_power-Pp49152': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00002034505208333333,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':159,
        'caption':'SemViT (P = 1/49152)'
    },
    'Yoo_power-Pp65536': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.0000152587890625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':193,
        'caption':'SemViT (P = 1/65536)'
    },
    'Yoo_power-Pp131072': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0.00000762939453125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':192,
        'caption':'SemViT with PAPR Loss (λ = 1/131072)'
    },
    
    ##### CLIP
    
    'Yoo_power-Mp2048-C0o75': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 0.75,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':186,
        'caption':'SemViT (C = 0.75)'
    },
    'Yoo_power-Mp2048-C0o875': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 0.875,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':192,
        'caption':'SemViT (C = 0.875)'
    },
    'Yoo_power-Mp2048-C0o9375': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 0.9375,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':200,
        'caption':'SemViT (C = 0.9375)'
    },
    'Yoo_power-Mp2048-C1': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':199,
        'caption':'SemViT (C = 1)'
    },
    'Yoo_power-Mp2048-C1o0625': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.0625,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':200,
        'caption':'SemViT (C = 1.0625)'
    },
    'Yoo_power-Mp2048-C1o125': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.125,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':194,
        'caption':'SemViT (C = 1.125)'
    },
    'Yoo_power-Mp2048-C1o25': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.25,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':191,
        'caption':'SemViT (C = 1.25)'
    },
    'Yoo_power-Mp2048-C1o5': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.5,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':199,
        'caption':'SemViT (C = 1.5)'
    },
    'Yoo_power-Mp2048-C1o75': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.75,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':189,
        'caption':'SemViT (C = 1.75)'
    },
    'Yoo_power-Mp2048-C2': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0.0004882812,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 2.0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':192,
        'caption':'SemViT (C = 2)'
    },
    
    'Yoo_power-C0o9375': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 0.9375,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':199,
        'caption':'SemViT (C = 0.9375)'
    },
    'Yoo_power-C1': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':199,
        'caption':'SemViT (C = 1)'
    },
    'Yoo_power-C1o25': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.25,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':170,
        'caption':'SemViT (C = 1.25)'
    },
    'Yoo_power-C1o75': {
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'papr_coeff'    : 0,
            'clip_limit'    : 1.75,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':199,
        'caption':'SemViT (C = 1.75)'
    },
    
    ### Various train SNR model ###
    
    'Yoo_base_train0':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 0,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':94,
        'caption':'SemViT train0'
    },      
    'Yoo_base_train5':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 5,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':97,
        'caption':'SemViT train 5dB'
    },      
    'Yoo_base_train15':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 15,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':94,
        'caption':'SemViT train 15dB'
    },      
    'Yoo_base_train20':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':87,
        'caption':'SemViT train 20dB'
    }, 
    'Yoo_base_train25':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 25,
            'num_symbols'   : 512,
            'filters'       : 256,
        },
        'epoch':99,
        'caption':'SemViT train 25dB'
    },  
    
    ## More clip
    
    'Yoo_power-NC1o03125':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'clip_limit'    : 1.03125,
        },
        'epoch':198,
        'caption':'SemViT clip 1.03125'
    },      
    'Yoo_power-nC1o03125':{
        'model_class':'SemViT_power',        
        'model_params':{
            'mean_coeff'    : 0,
            'std_coeff'     : 0,
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'clip_limit'    : 1.03125,
        },
        'epoch':190,
        'caption':'SemViT clip 1.03125'
    },  
    
    ## OFDM PAPR Reduction
    'SemViT_Pp1024':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.0009765625
        },
        'epoch':200,
        'caption':'SemViT PAPR (K = 1/1024)'
    },
    'SemViT_Pp2048':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.00048828125
        },
        'epoch':199,
        'caption':'SemViT PAPR (K = 1/2048)'
    },
    'SemViT_Pp4096':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.000244140625
        },
        'epoch':196,
        'caption':'SemViT PAPR (K = 1/4096)'
    },
    'SemViT_Pp8192':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.0001220703125
        },
        'epoch':192,
        'caption':'SemViT PAPR (K = 1/8192)'
    },
    'SemViT_Pp16384':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.00006103515625
        },
        'epoch':172,
        'caption':'SemViT PAPR (K = 1/16384)'
    },
    'SemViT_Pp32768':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 10,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.000030517578125
        },
        'epoch':199,
        'caption':'SemViT PAPR (K = 1/32768)'
    },
    #20dB
    'SemViT_20_Pp1024':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.0009765625
        },
        'epoch':200,
        'caption':'SemViT PAPR (K = 1/1024)'
    },
    'SemViT_20_Pp2048':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.00048828125
        },
        'epoch':196,
        'caption':'SemViT PAPR (K = 1/2048)'
    },
    'SemViT_20_Pp4096':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.000244140625
        },
        'epoch':199,
        'caption':'SemViT PAPR (K = 1/4096)'
    },
    'SemViT_20_Pp8192':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.0001220703125
        },
        'epoch':199,
        'caption':'SemViT PAPR (K = 1/8192)'
    },
    'SemViT_20_Pp16384':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.00006103515625
        },
        'epoch':190,
        'caption':'SemViT PAPR (K = 1/16384)'
    },
    'SemViT_20_Pp32768':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.000030517578125
        },
        'epoch':197,
        'caption':'SemViT PAPR (K = 1/32768)'
    },
    'SemViT_20_Pp65536':{
        'model_class':'SemViT_OFDM_PAPR', 
        'model_params':{
            'snrdB'         : 20,
            'num_symbols'   : 512,
            'filters'       : 256,
            'papr_coeff'    : 0.0000152587890625
        },
        'epoch':58,
        'caption':'SemViT PAPR (K = 1/65536)'
    },
} 

    
def getModelConfigs(root='logs/model_configs.json'):
    # with open('logs/model_configs.json', 'w') as f:
    #     json.dump(model_parameters, f)
    return model_parameters