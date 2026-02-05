import numpy as np
import os
import gc

nps_classes = [
    'Cathinone analogues',
    'Cannabinoid analogues',
    'Phenethylamine analogues',
    'Piperazine analogues',
    'Tryptamine analogues',
    'Fentanyl analogues'
]

def random_position(free_cells):
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(y), float(x)

def clean_tmp_files():
    tmp_files = [
        "/tmp/C1.npy",
        "/tmp/binary_map.npy"
    ]

    for f in tmp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"[cleanup] Cannot remove {f}: {e}")

    gc.collect()