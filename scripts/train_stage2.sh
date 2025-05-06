# ##### MF Dataset
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python train/train_stage2.py --config ./configs/stage2_MFD_train.yaml --stage1_model_name stage1_model

##### KP Dataset
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python train/train_stage2.py --config ./configs/stage2_KPD_train.yaml --stage1_model_name stage1_model