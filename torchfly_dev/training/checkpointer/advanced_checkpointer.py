import os
import glob
import logging
import torch
import time
from collections import deque
from .base_checkpointer import BaseCheckpointer

from typing import Union, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class My_list():
    def __init__(self, num, allowed_time):  
        
        self.num = num
        self.allowed_time = allowed_time
        
        self.safe_list = []
        self.temp_list = deque(maxlen=num)
        self.time_list = deque(maxlen=num)
        
    def append(self, x, t):
        
        if len(self.temp_list)<self.num:
            # then store this value and no element kicked out  
            self.temp_list.append(x)
            self.time_list.append(t)
            out = None
        else:    
            
            how_long = t - self.time_list[0]
            
            if how_long > self.allowed_time:
                # then the oldest one in temp_list is safe, and put it into save list
                oldest = self.temp_list.popleft()
                self.safe_list.append(oldest)
                self.temp_list.append(x)
                self.time_list.append(t) # we do not need timestamp of this oldest one anymore 
                out = None
            else:
                # the oldest one in temp_list should be removed  
                out = self.temp_list.popleft()
                self.temp_list.append(x)
                self.time_list.append(t)
            
        return out
    
    def __getitem__(self, item):
        all_list = self.safe_list + list(self.temp_list)
        return all_list[item]


class AdavancedCheckpointer(BaseCheckpointer):
    """
    Simple Checkpointer implements the basic functions
    """
    def __init__(self, num_checkpoints_to_keep=1000, keep_checkpoint_every_num_seconds=3600, storage_dir="Checkpoints"):
        
        self.storage_dir = storage_dir
        self.checkpoints_list = My_list(num_checkpoints_to_keep, keep_checkpoint_every_num_seconds)

        # initialization
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_checkpoint(self, stamp:str, state: Dict[str, Any]) -> None:
        """
        Args:
            stamp: A string to identify the checkpoint. It can just be the epoch number
            state: A dictionary to store all necessary information for later restoring

        """
        
        checkpoint_path = os.path.join(self.storage_dir,f"{stamp}_state.pth")        
        t = time.time()
        
        # check if any old checkpoints need to be removed 
        need_remove_checkpoint = self.checkpoints_list.append(checkpoint_path, t)
        
        # if so remove it 
        if need_remove_checkpoint is not None:
            os.remove(need_remove_checkpoint)

        # store the lastest one
        torch.save(state, checkpoint_path)

        
    def restore_checkpoint(self, search_method=None):
        """
        Args:
            search_method: a Callable to find the wanted checkpoint path
        """
         
        if search_method is None:
            # if not specified then the last one is lastest checkpoint
            checkpoint_path = self.checkpoints_list[-1]
        else:
            checkpoint_path = search_method()
            
        # map to the cpu first instead of error
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        return checkpoint
