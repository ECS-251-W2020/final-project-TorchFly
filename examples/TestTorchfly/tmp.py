import torch

class Test:
    def __init__(self):
        self.value = 5
    
    def train(self):
        torch.multiprocessing.spawn(self.dist, args=(), nprocs=8)
#         torch.distributed.barrier()
    
    def dist(self, rank):
        print(rank)
        with open(f"Checkpoints/{rank}.txt", "w") as f:
            f.write(str(self.value + rank))
            
        return 0