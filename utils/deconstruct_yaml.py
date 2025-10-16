import copy
import itertools

import yaml
from utils.get_path import get_project_path

def get_config_from_yaml(file_name):
    if file_name is not None:
        with open(get_project_path() + '/configs/' + file_name, 'r', encoding='utf-8') as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
            return result

def mix_config_parser(args, config):
    common_config = get_config_from_yaml("common_config.yaml")
    config.update(common_config)
    hyper_parms = []
    for key, value in config.items():
        if isinstance(value, list):
            hyper_parms.append([key, value])
        else:
            setattr(args, key, value)

    a = [item[-1] for item in hyper_parms]
    hyper_parms_list = list(itertools.product(*a))
    args_list = []
    for hyper_parm in hyper_parms_list:
        arg = copy.deepcopy(args)
        for i in range(len(hyper_parm)):
            key = hyper_parms[i][0].split("_list")[0]
            value = hyper_parm[i]
            setattr(arg, key, value)
        args_list.append(arg)

    return args_list