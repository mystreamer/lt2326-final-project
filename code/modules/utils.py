import torch

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

def detect_platform(cuda_num):
    if torch.cuda.is_available():
        print("cuda is available")
        return f'cuda:{cuda_num}'
    elif torch.backends.mps.is_available():
        print("mps is available")
        return 'mps'
    else:
        print("cpu is available")
        return 'cpu'

if __name__ == "__main__":
    print("Nothin to do yet...")