##### MF Dataset
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python train/train_stage1.py --config ./configs/stage1_MFD_train.yaml

##### KP Dataset
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python train/train_stage1.py --config ./configs/stage1_KPD_train.yaml 
