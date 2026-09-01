import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_tfrecord_type_probe import parse_example
from preprocess_target_centric_prediction import iter_tfrecord_records

sample_files = sorted(glob.glob(r"E:\motion_data\rideflux_91f_full\rideflux\**\*.tfrecord", recursive=True))
print(f"Found {len(sample_files)} tfrecord files.")
first_file = sample_files[0]
print(f"Testing sample file: {first_file}")

for i, raw_bytes in enumerate(iter_tfrecord_records(first_file)):
    feat = parse_example(raw_bytes)
    print(f"Record {i}: parsed {len(feat)} features successfully.")
    for k in ['state/past/x', 'state/current/x', 'state/future/x', 'state/type', 'roadgraph_samples/xyz']:
        if k in feat:
            kind, vals = feat[k]
            print(f"  - {k}: {kind}, count={len(vals)}")
    break
