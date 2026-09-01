@echo off
echo ===============================================================================
echo  [Step 4] Training Motion Prediction V17 X-Large 45.2M Graph Model (30 Epochs)
echo ===============================================================================
python train_motion_prediction_v17.py ^
    --cache-root "data\processed\prediction_pt_85k_v2_cache_v13" ^
    --out-dir "checkpoints\v17_xlarge" ^
    --batch-scenes 24 ^
    --epochs 30 ^
    --lr 1.5e-4 ^
    --hidden 768 ^
    --nhead 12 ^
    --dropout 0.1 ^
    --amp bf16 ^
    --workers 8
pause
