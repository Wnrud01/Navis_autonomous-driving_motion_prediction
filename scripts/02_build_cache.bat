@echo off
echo ===============================================================================
echo  [Step 2] Building High-Speed GPU Collate Cache
echo ===============================================================================
python data_tools\cache_collate_v13.py --data-root "data\processed\prediction_pt_85k_v2" --out-dir "data\processed\prediction_pt_85k_v2_cache_v13" --workers 12
pause
