import os
from copy import deepcopy

def divide_sweep(original_cfg):
	base_config = {}
	sweep_config = {"method": original_cfg["method"], "parameters":{}}
	del original_cfg["method"]

	split_config(original_cfg,base_config,sweep_config)

	return base_config, sweep_config
	

def split_config(cfg, nested_base_config, sweep_config, nested_key=""):
	for k,v in cfg.items():
		if "+"==k[0]: #is a sweep_parameter
			real_key = k[1:]
			sweep_config["parameters"][".".join(filter(None, [nested_key,real_key]))] = {"values":v}
			nested_base_config[real_key] = None
		else:
			if isinstance(v,dict):
				nested_base_config[k] = {}
				split_config(v, nested_base_config[k], sweep_config, ".".join(filter(None, [nested_key,k])))
			else:
				nested_base_config[k] = v


def merge_configs(base_cfg,sweep_cfg):
	complete_cfg = deepcopy(base_cfg)

	for nested_k,v in sweep_cfg.items():
		iterate_keys(complete_cfg,nested_k.split("."),v)
		
	return complete_cfg

def iterate_keys(nested_dict,k_list,v):
	if len(k_list) == 1:
		nested_dict[k_list[0]] = v
	else:
		iterate_keys(nested_dict[k_list[0]],k_list[1:],v)

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __init__(self,dct):
		super().__init__(dct)
		for k,v in dct.items():
			if isinstance(v,dict):
				v = dotdict(v)



