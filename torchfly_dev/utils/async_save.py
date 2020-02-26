import ray
import torch
import numpy as np
from collections import OrderedDict

def get_cpu_state_dict(states: OrderedDict) -> OrderedDict:
    for k in states:
        states[k] = states[k].cpu().numpy()
    return states

@ray.remote
def _async_save(states, filename):
    for k in states:
        states[k] = torch.from_numpy(states[k])
        
    torch.save(states, filename)
    return 0

def async_save(model_states: OrderedDict, filename):
    model_states = get_cpu_state_dict(model_states)
    ray_obj = _async_save.remote(model_states, filename)
    return ray_obj

def check_async_status(ray_obj):
    return ray.get(ray_obj)