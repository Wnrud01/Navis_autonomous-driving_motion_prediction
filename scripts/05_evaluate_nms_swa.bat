@echo off
echo ===============================================================================
echo  [Step 5] Evaluating SWA Checkpoint Averaging + Soft Density Trajectory NMS
echo ===============================================================================
python evaluate_v16_swa.py ^
    --ckpt-dir "checkpoints\v16_twostage" ^
    --cache-root "data\processed\prediction_pt_85k_v2_cache_v13" ^
    --batch-scenes 64
pause
