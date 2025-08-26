mkdir logs/sym512_train10_base
CUDA_VISIBLE_DEVICES=0 nohup python train.py sym512_train10_base --train_snrdB 10 --epochs 200 &> logs/sym512_train10_base/nohup.out &

mkdir logs/indiv_norm
CUDA_VISIBLE_DEVICES=0 nohup python train.py indiv_norm --model JSCC --train_snrdB 10 --epochs 200 &> logs/indiv_norm/nohup.out &

mkdir logs/learn_norm_L1
CUDA_VISIBLE_DEVICES=0 nohup python train.py learn_norm_L1 --model JSCC_norm --lamda 1 --train_snrdB 10 --epochs 200 &> logs/learn_norm_L1/nohup.out &

mkdir logs/learn_norm_L16
CUDA_VISIBLE_DEVICES=1 nohup python train.py learn_norm_L16 --model JSCC_norm --lamda 16 --train_snrdB 10 --epochs 200 &> logs/learn_norm_L16/nohup.out &

mkdir logs/learn_norm_mult_L1
CUDA_VISIBLE_DEVICES=0 nohup python train.py learn_norm_mult_L1 --model JSCC_norm_mult --lamda 1 --train_snrdB 10 --epochs 200 &> logs/learn_norm_mult_L1/nohup.out &

mkdir logs/learn_norm_mult_L1000
CUDA_VISIBLE_DEVICES=1 nohup python train.py learn_norm_mult_L1000 --model JSCC_norm_mult --lamda 1000 --train_snrdB 10 --epochs 200 &> logs/learn_norm_mult_L1000/nohup.out &

mkdir logs/learn_norm_L64
CUDA_VISIBLE_DEVICES=0 nohup python train.py learn_norm_L64 --model JSCC_norm --lamda 64 --train_snrdB 10 --epochs 200 &> logs/learn_norm_L64/nohup.out &

mkdir logs/learn_norm_L64_LRE3
CUDA_VISIBLE_DEVICES=1 nohup python train.py learn_norm_L64_LRE3 --model JSCC_norm --lamda 64 --train_snrdB 10 --epochs 200 --learning_rate 0.001 &> logs/learn_norm_L64_LRE3/nohup.out &

mkdir logs/learn_norm_L256
CUDA_VISIBLE_DEVICES=1 nohup python train.py learn_norm_L256 --model JSCC_norm --lamda 256 --train_snrdB 10 --epochs 200 &> logs/learn_norm_L256/nohup.out &

#####

experiment=learn_power_Mp16_Sp0
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_power --mean_coeff 0.0625 --std_coeff 0 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=learn_power_Mp16_Sp64
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_power --mean_coeff 0.0625 --std_coeff 0.015625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=learn_power_Mp16_Sp256
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_power --mean_coeff 0.0625 --std_coeff 0.00390625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=learn_power_Mp16_Sp1024
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_power --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

#####

experiment=JSCC_power_powL1_powStd
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_ampL2_powVar
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_ampL2_ampStd
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_ampL2_ampVar
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_powL1_ampStd
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_powL1_ampVar
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_ampL2_powStd
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_powL1_powVar
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model $experiment --mean_coeff 0.0625 --std_coeff 0.0009765625 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

#####

experiment=JSCC_power_ampL2_ampVar_Lp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_power_ampL2_ampVar --mean_coeff 0.0625 --std_coeff 0.00048828125 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_power_ampL2_ampVar_Lp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_power_ampL2_ampVar --mean_coeff 0.0625 --std_coeff 0.00024414062 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

#####

experiment=JSCC_usrp_powL1_M0
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0 --precision 5 --clip_coeff 4 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

experiment=JSCC_usrp_powL1_Mp256
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0.00390625 --precision 5 --clip_coeff 4 --train_snrdB 10 --epochs 200 &> logs/$experiment/nohup.out &

##### Power L1 mean loss, Pow Std loss, Clip MSE Loss

experiment=JSCC-usrp_M0_S0_Cp16-b4-l4
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0 --std_coeff 0 --clip_coeff 0.0625 --precision 4 --clip_limit 4 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_M0_S0_Cp64-b4-l4
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0 --std_coeff 0 --clip_coeff 0.015625 --precision 4 --clip_limit 4 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_M0_S0_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0 --std_coeff 0 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_M64_S4096_Cp16-b4-l4
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0.015625 --std_coeff 0.000244140625 --clip_coeff 0.0625 --precision 4 --clip_limit 4 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_M64_S0_Cp64-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0.015625 --std_coeff 0 --clip_coeff 0.015625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_M64_S4096_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0.015625 --std_coeff 0.000244140625 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_Mp64_S0_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0.015625 --std_coeff 0 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp_Mp64_S16384_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp --mean_coeff 0.015625 --std_coeff 0.00006103515625 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

##### Easy USRP Model

experiment=JSCC-usrp-sim_M0_S0_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp_simple --mean_coeff 0 --std_coeff 0 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp-sim_M64_S4096_Cp16-b4-l4
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp_simple --mean_coeff 0.015625 --std_coeff 0.000244140625 --clip_coeff 0.0625 --precision 4 --clip_limit 4 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp-sim_Mp64_S0_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model JSCC_usrp_simple --mean_coeff 0.015625 --std_coeff 0 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

experiment=JSCC-usrp-sim_Mp64_S16384_Cp16-b4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model JSCC_usrp_simple --mean_coeff 0.015625 --std_coeff 0.00006103515625 --clip_coeff 0.0625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &


##### Yoo model
experiment=Yoo-base
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT &> logs/$experiment/nohup.out &

experiment=Yoo-usrp_Mp64_Sp16384_Cb4-l2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT --mean_coeff 0.015625 --std_coeff 0.00006103515625 --precision 4 --clip_limit 2 &> logs/$experiment/nohup.out &

##### REAL Channel Model

experiment=real_channel-M0-S0-Cl8-Fao15_ho05
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_real --mean_coeff 0 --std_coeff 0 --clip_limit 8 --amp_fluc 0.15 --phase_fluc 0.05 --ckpt "logs/Yoo-base/weights/epoch_195" --epoch 10 &> logs/$experiment/nohup.out &

experiment=real_channel_indiv-M0-S0-Cl8-Fao15_ho05
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_real --mean_coeff 0 --std_coeff 0 --clip_limit 8 --amp_fluc 0.15 --phase_fluc 0.05 --ckpt "logs/Yoo-base/weights/epoch_195" --epoch 10 &> logs/$experiment/nohup.out &

experiment=rice_channel_indiv-M0-S0-Cl8-Fo13
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_rician --mean_coeff 0 --std_coeff 0 --clip_limit 8 --shape_param 0.13 --ckpt "logs/Yoo-base/weights/epoch_195" --epoch 10 &> logs/$experiment/nohup.out &.out &

### Back To Power Normalization

experiment=Yoo_base
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --ckpt "logs/Yoo-base/final" --initial_epoch 200 --epoch 400 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp64
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.015625 --ckpt "logs/Yoo-base/final" --epoch 200 &> logs/$experiment/nohup.out &

###

experiment=Yoo_power-Mp64
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.015625 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --initial_epoch 200 --epoch 300 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp16
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.0625 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp256
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00390625 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

### Standard Deviation

experiment=Yoo_power-Sp16384
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0.00006103515625 --ckpt "logs/Yoo-base/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Sp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0.000244140625 --ckpt "logs/Yoo-base/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Sp1024
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0.0009765625 --ckpt "logs/Yoo-base/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Sp256
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0.00390625 --ckpt "logs/Yoo-base/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Sp64
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0.015625 --ckpt "logs/Yoo-base/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Sp16
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0.0625 --ckpt "logs/Yoo-base/final" --epoch 100 &> logs/$experiment/nohup.out &

### For Fair Comparison

experiment=Yoo_base
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --ckpt "logs/Yoo_base/final" --initial_epoch 500 --epoch 700 &> logs/$experiment/nohup.out &

### Strict Mean

experiment=Yoo_power-Mp4
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.25 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp8
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.125 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp16
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.0625 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp32
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.03125 --std_coeff 0 --ckpt "logs/Yoo_power-Mp64/final" --epoch 100 &> logs/$experiment/nohup.out &

### PAPR LOSS

experiment=Yoo_power-Mp2048-Pp16
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.0625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp32
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.03125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp64
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.015625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp128
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.0078125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp256
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.00390625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp512
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.001953125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp1024
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.0009765625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.00048828125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.000244140625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp8192
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.0001220703125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp16384
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.00006103515625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp24576
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.00004069010416666667 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp32768
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.000030517578125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp49152
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.00002034505208333333 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp65536
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.0000152587890625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-Pp131072
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --papr_coeff 0.00000762939453125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

# ONLY PAPR

experiment=Yoo_power-Pp16384
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.00006103515625 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Pp24576
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.00004069010416666667 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Pp32768
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.000030517578125 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Pp49152
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.00002034505208333333 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Pp65536
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.0000152587890625 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Pp131072
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.00000762939453125 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &


### CLIP for PAPR

experiment=Yoo_power-Mp2048-C1o25
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 1.25 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C1o5
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 1.5 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C1o75
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 1.75 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 2.0 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C0o75
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 0.75 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C0o875
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 0.875 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C0o9375
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 0.9375 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C1
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 1.0 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C1o0625
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 1.0625 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-Mp2048-C1o125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0.00048828125 --std_coeff 0 --clip_limit 1.125 --ckpt "logs/Yoo_power-Mp2048/final" --epoch 200 &> logs/$experiment/nohup.out &

### Only Clip

experiment=Yoo_power-C1o25
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.25 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-C1o75
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.75 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-C0o9375
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 0.9375 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-C1
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.0 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &



# <======================================== USRP Like Models ========================================> #

### simple

experiment=USRP_base
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type simple --ckpt "logs/USRP_initial/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_base_fromclip
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type simple --ckpt "logs/USRP_initial_clip/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_base_meanpapr
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type simple --ckpt "logs/USRP_initial_meanpapr/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_base_fromclip
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type simple --ckpt "logs/USRP_base_fromclip/final" --initial_epoch 200 --epoch 400 &> logs/$experiment/nohup.out &

### penalty

experiment=USRP_penalty-Cp64-scratch
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type penalty --clip_coeff 0.015625 --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_penalty-Cp512-scratch
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type penalty --clip_coeff 0.001953125 --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_penalty-Cp64-ckpt
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type penalty --clip_coeff 0.015625 --ckpt "logs/USRP_base_meanpapr/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_penalty-Cp512-ckpt
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type penalty --clip_coeff 0.001953125 --ckpt "logs/USRP_base_meanpapr/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_penalty-Cp2048-ckpt
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type penalty --clip_coeff 0.00048828125 --ckpt "logs/USRP_base_meanpapr/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_penalty-Cp8192-ckpt
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 16 --clipping_type penalty --clip_coeff 0.0001220703125 --ckpt "logs/USRP_base_meanpapr/final" --epoch 200 &> logs/$experiment/nohup.out &


# <======================================== Various SNR Model ========================================> #


experiment=Yoo_base_train0
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --ckpt "logs/Yoo_base/final" --epoch 100 --train_snrdB 0 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train5
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --ckpt "logs/Yoo_base/final" --epoch 100 --train_snrdB 5 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train15
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --ckpt "logs/Yoo_base/final" --epoch 100 --train_snrdB 15 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train20
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --ckpt "logs/Yoo_base/final" --epoch 100 --train_snrdB 20 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train25
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --ckpt "logs/Yoo_base/final" --epoch 100 --train_snrdB 25 &> logs/$experiment/nohup.out &


### Only Clip --Upgrade

experiment=Yoo_power-NC1o03125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.03125 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC1o0625
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.0625 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC1o125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.125 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC1o1875
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.1875 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC1o25
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.25 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC1o5
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.5 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC1o75
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.75 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-NC2
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 2.0 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

### Only Clip --Upgrade More

experiment=Yoo_power-nC1o03125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.03125 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-nC1o0625
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.0625 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-nC1o125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.125 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-nC1o1875
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.1875 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-nC1o25
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.25 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

experiment=Yoo_power-nC1o5
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --clip_limit 1.5 --ckpt "logs/Yoo_base/final" --epoch 200 &> logs/$experiment/nohup.out &

### For Paper

experiment=Yoo_power-Pp16384-train20
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --mean_coeff 0 --std_coeff 0 --papr_coeff 0.00006103515625 --ckpt "logs/Yoo_power-Pp16384/final" --epoch 200 --train_snrdB 20 &> logs/$experiment/nohup.out &


### Final Quantization
experiment=USRP_quantize_0dB
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_usrp_freezeEnc --freeze_encoder --Rx_precision 16 --attenuation 0.001 --normalizer 185 --train_snrdB 0 --ckpt "logs/USRP_quantize_0dB/init" --epoch 200 &> logs/$experiment/nohup.out &

experiment=USRP_quantize_0dB_mimic
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_usrp --Rx_precision 4 --attenuation 0.25 --normalizer 12 --train_snrdB 0 --ckpt "logs/USRP_quantize_0dB/init" --epoch 200 &> logs/$experiment/nohup.out &

### Quant mimic


### Channel learnable model

experiment=SemViT_rician_zf
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_rician --shape_param 50 --channel_type ZF --train_snrdB 10 --epoch 500 &> logs/$experiment/nohup.out &
experiment=SemViT_rician_dnn
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_rician --shape_param 50 --channel_type DNN --train_snrdB 10 --epoch 500 &> logs/$experiment/nohup.out &

experiment=SemViT_SNRAdap_div
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_SNRAdap --channel_type DIV --train_snrdB 10 --epoch 500 &> logs/$experiment/nohup.out &
experiment=SemViT_SNRAdap_dnn
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_SNRAdap --channel_type DNN --train_snrdB 10 --epoch 500 &> logs/$experiment/nohup.out &

### OFDM PAPR model
experiment=SemViT_P0
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --train_snrdB 10 --papr_coeff 0 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --train_snrdB 10 --papr_coeff 0.00024414062 --epoch 100 &> logs/$experiment/nohup.out &

experiment=SemViT_Pp1024
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base/final" --papr_coeff 0.0009765625 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_Pp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base/final" --papr_coeff 0.00048828125 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base/final" --papr_coeff 0.000244140625 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_Pp8192
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base/final" --papr_coeff 0.0001220703125 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_Pp16384
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base/final" --papr_coeff 0.00006103515625 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_Pp32768
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base/final" --papr_coeff 0.000030517578125 --epoch 200 &> logs/$experiment/nohup.out &

# 20dB
experiment=SemViT_20_Pp1024
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.0009765625 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_20_Pp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.00048828125 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_20_Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.000244140625 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_20_Pp8192
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.0001220703125 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_20_Pp16384
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.00006103515625 --epoch 200 &> logs/$experiment/nohup.out &
experiment=SemViT_20_Pp32768
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.000030517578125 --epoch 200 &> logs/$experiment/nohup.out &

experiment=SemViT_20_Pp65536
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train20/final" --train_snrdB 20 --papr_coeff 0.0000152587890625 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_15_Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train15/final" --train_snrdB 15 --papr_coeff 0.000244140625 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_15_Pp8192
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train15/final" --train_snrdB 15 --papr_coeff 0.0001220703125 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_15_Pp16384
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train15/final" --train_snrdB 15 --papr_coeff 0.00006103515625 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_15_Pp32768
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train15/final" --train_snrdB 15 --papr_coeff 0.000030517578125 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_5_Pp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train5/final" --train_snrdB 5 --papr_coeff 0.00048828125 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_5_Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train5/final" --train_snrdB 5 --papr_coeff 0.000244140625 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_5_Pp8192
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train5/final" --train_snrdB 5 --papr_coeff 0.0001220703125 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_0_Pp2048
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=1 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train0/final" --train_snrdB 0 --papr_coeff 0.00048828125 --epoch 100 &> logs/$experiment/nohup.out &
experiment=SemViT_0_Pp4096
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_OFDM_PAPR --ckpt "logs/Yoo_base_train0/final" --train_snrdB 0 --papr_coeff 0.000244140625 --epoch 100 &> logs/$experiment/nohup.out &





# <======================================== Various SPP Model (250811) ========================================> #

experiment=Yoo_base_train0_spp125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --epoch 500 --train_snrdB 0 --num_symbols 256 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train0_spp5
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --epoch 500 --train_snrdB 0 --num_symbols 1024 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train20_spp125
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --epoch 500 --train_snrdB 20 --num_symbols 256 &> logs/$experiment/nohup.out &

experiment=Yoo_base_train20_spp5
mkdir logs/$experiment
CUDA_VISIBLE_DEVICES=0 nohup python train.py $experiment --model SemViT_power --epoch 500 --train_snrdB 20 --num_symbols 1024 &> logs/$experiment/nohup.out &
