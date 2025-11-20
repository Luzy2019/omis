import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from pretraining.utils import train_opponent_index, test_opponent_index

if __name__ == "__main__":
    SEED = 777
    np.random.seed(SEED)
    test_modes = ["seen", "unseen", "mix"]
    num_test = 1200
    switch_intervals = [2, 5, 10, 20, "D"]
    for tm in test_modes:
        for si in switch_intervals:
            if tm == "seen":
                test_oppo_policy = train_opponent_index
            elif tm == "unseen":
                test_oppo_policy = test_opponent_index
            elif tm == "mix":
                test_oppo_policy = train_opponent_index+test_opponent_index
            
            print("len(test_oppo_policy)", len(test_oppo_policy))
            if si != "D":
                length = num_test // si
                
                num_repeat = length // len(test_oppo_policy)
                test_oppo_seq = np.repeat(np.arange(len(test_oppo_policy)), num_repeat) 
                np.random.shuffle(test_oppo_seq)
                
                print(test_oppo_seq[:50])
                print(test_oppo_seq.shape)
                print(np.unique(test_oppo_seq, return_counts=True))
            else:
                test_oppo_seq = []
                while len(test_oppo_seq) < num_test:
                    length = np.random.choice(switch_intervals[:4])
                    oppo_idx = np.random.randint(len(test_oppo_policy))
                    oppo_seq = [oppo_idx for _ in range(length)]
                    test_oppo_seq += oppo_seq
                test_oppo_seq = np.array(test_oppo_seq[:num_test])
                
                print(test_oppo_seq[:50])
                print(test_oppo_seq.shape)

            with open(f"{tm}_oppo_switch_{si}.npy", 'wb') as f:
                np.save(f, test_oppo_seq)