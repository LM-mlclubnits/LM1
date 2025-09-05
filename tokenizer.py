!pip install tiktoken
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
enc=tiktoken.get_encoding("gpt2")
def process(example):
    ids=enc.encode_ordinary(example['text'])
    out={'ids': ids,
         'len': len(ids)
        }
    return out
if not os.path.exists("train.bin"):
    tokenized=ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,)
    for split, dset in tokenized.items():
        arr_len=np.sum(dset['len'], dtype=np.uint64)
        filename=f'{split}.bin'
        dtype=np.uint16
        arr=np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        tokens_per_batch=int(1e6)
        total_batches=max(1, arr_len // tokens_per_batch)
        idx=0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch=dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch=np.concatenate(batch['ids'])
            arr[idx:idx+len(arr_batch)]=arr_batch
            idx+=len(arr_batch)
        arr.flush()
