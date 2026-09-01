Set-Location 'E:\motion_prediction'
$log = (Join-Path (Get-Location) '.\checkpoints\v8_hardcls\console.log')
$err = (Join-Path (Get-Location) '.\checkpoints\v8_hardcls\console.err.log')
$pythonArgs = @(
  '.\train_motion_prediction_v8.py',
  '--data-root', 'E:\motion_planning\data\processed\prediction_pt_85k',
  '--out-dir', '.\checkpoints\v8_hardcls',
  '--resume-ckpt', '.\checkpoints\v6_temporal\best_error_score.pth',
  '--epochs', '20',
  '--batch-scenes', '32',
  '--workers', '8',
  '--prefetch', '4',
  '--lr', '1.0e-4',
  '--tau-cls', '0.30',
  '--weight-fde', '0.75',
  '--weight-cls', '0.8',
  '--weight-hard-cls', '0.5',
  '--amp', 'bf16'
)
$p = Start-Process -FilePath 'python' -ArgumentList $pythonArgs -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
Write-Output "V8_PID=$($p.Id)"
Start-Sleep -Seconds 8
Get-Content $log -Tail 24
