@echo off
echo ===============================================================================
echo  [Step 1] Converting Raw TFRecords to Preprocessed .pt Packs
echo ===============================================================================
python data_tools\preprocess_85k_v2.py --raw-root "data\raw" --out-root "data\processed\prediction_pt_85k_v2" --workers 12
pause
