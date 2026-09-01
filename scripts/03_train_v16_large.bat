@echo off
echo ===============================================================================
echo  [Step 3] Training Motion Prediction V16 Large Two-Stage Model (30 Epochs)
echo ===============================================================================
python train_motion_prediction_v16.py ^
    --cache-root "data\processed\prediction_pt_85k_v2_cache_v13" ^
    --out-dir "checkpoints\v16_twostage" ^
    --batch-scenes 32 ^
    --epochs 30 ^
    --lr 2e-4 ^
    --hidden 256 ^
    --amp bf16 ^
    --workers 8
pause
