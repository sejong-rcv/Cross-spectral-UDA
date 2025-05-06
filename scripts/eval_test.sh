# ### MF Dataset
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python ./train/train_stage1.py --config ./configs/stage2_MFD_train.yaml --test_mode 

# ### KP Dataset
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python ./train/train_stage1.py --config ./configs/stage2_KPD_train.yaml --test_mode 