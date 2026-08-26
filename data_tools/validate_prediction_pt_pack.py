#!/usr/bin/env python3
"""Read-only validation for one target-centric prediction pack."""
from __future__ import annotations

import argparse
import glob
import json
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
paths = sorted(glob.glob(os.path.join(args.data_dir, 'train', '*.pt')) + glob.glob(os.path.join(args.data_dir, 'val', '*.pt')))
if not paths:
    raise FileNotFoundError('No prediction packs found')
path = paths[len(paths) // 2]
pack = torch.load(path, map_location='cpu', weights_only=False)
assert pack['schema_version'] == 'target_centric_prediction_v1'
assert len(pack['windows']) == 3
report = {
    'sample_path': os.path.relpath(path, args.data_dir),
    'schema_version': pack['schema_version'],
    'scenario_id': pack['scenario_id'],
    'static_shapes': {k: list(v.shape) for k, v in pack['static'].items() if hasattr(v, 'shape')},
    'windows': [],
}
for window in pack['windows']:
    inputs = window['inputs']
    targets = window['targets']
    assert list(inputs['agent_history_world'].shape[-2:]) == [11, 5]
    assert list(inputs['agent_history_valid'].shape[-1:]) == [11]
    assert list(targets['future_xy_world'].shape[-2:]) == [80, 2]
    assert list(targets['future_valid'].shape[-1:]) == [80]
    assert all('future' not in key for key in inputs), 'future feature leaked into inputs'
    report['windows'].append({
        'window_index': int(window['window_index']),
        'anchor_global_step': int(window['anchor_global_step']),
        'target_count': int(targets['target_rows'].numel()),
        'agent_history_shape': list(inputs['agent_history_world'].shape),
        'future_xy_shape': list(targets['future_xy_world'].shape),
        'target_types': sorted(set(int(x) for x in targets['target_types'].tolist())),
    })
with open(args.out, 'w', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
