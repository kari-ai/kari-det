import sys
import contextlib
import re
from difflib import get_close_matches
from types import SimpleNamespace
from typing import Dict, List, Union

from kari.det.utils import (DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, DEFAULT_CFG, LOGGER, ROOT, yaml_load, colorstr, checks)

CLI_HELP_MSG = \
    """
    Usage: kari-det MODE ARGS

    MODE (required) is one of [train, val, predict]
    ARGS (optional) are any number of custom 'arg=value' pairs like 'model=yolov8n.pt' that override defaults.

    Examples:
    1. Train a detection model for 10 epochs with an initial learning rate of 0.01:
       kari-det train model=yolov8n.pt data=data/kari_obj_hbb.yaml epochs=10 lr0=0.01
    
    2. Val a pretrained detection model at batch-size 1 and image size 640:
       kari-det val model=yolov8n.pt data=data/kari_obj_hbb.yaml batch-size=1 img-size=640
       
    3. Run special commands:
       kari-det help
       kari-det checks
       kari-det version
    """

MODES = 'train', 'val', 'predict'

def entrypoint(debug=''):
    args = (debug.split(' ') if debug else sys.argv)[1:]
    if not args:
        LOGGER.info(CLI_HELP_MSG)
        return
    
    special = {
        'help': lambda: LOGGER.info(CLI_HELP_MSG),
        'checks': checks.check_kari_det,
    }

    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in MODES}, **special}
    overrides = {}

    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith('--'):
            LOGGER.warning(f"WARNING ⚠️ '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(','):
            LOGGER.warning(f"WARNING ⚠️ '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if '=' in a:
            try:
                re.sub(r' *= *', '=', a)  # remove spaces around equals sign
                k, v = a.split('=', 1)  # split on first '=' sign
                assert v, f"missing '{k}' value"
                if k == 'cfg':  # custom.yaml passed
                    LOGGER.info(f'Overriding {DEFAULT_CFG_PATH} with {v}')
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != 'cfg'}
                else:
                    if v.lower() == 'none':
                        v = None
                    elif v.lower() == 'true':
                        v = True
                    elif v.lower() == 'false':
                        v = False
                    else:
                        with contextlib.suppress(Exception):
                            v = eval(v)
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_cfg_mismatch(full_args_dict, {a: ''}, e)

        elif a in MODES:
            overrides['mode'] = a
        elif a in special:
            special[a]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(f"'{colorstr('red', 'bold', a)}' is a valid kari-det argument but is missing an '=' sign "
                              f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}")
        else:
            check_cfg_mismatch(full_args_dict, {a: ''})

    # Check for keys
    check_cfg_mismatch(full_args_dict, overrides)

    # Mode
    mode = overrides.get('mode', None)
    if mode is None:
        mode = DEFAULT_CFG.mode or 'predict'
        LOGGER.warning(f"WARNING ⚠️ 'mode' is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        if mode not in ('checks', checks):
            raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")
        checks.check_kari_det()
        return
    
    # Model
    model = overrides.pop('model', DEFAULT_CFG.model)
    if model is None:
        model = 'yolov8n.pt'
        LOGGER.warning(f"WARNING ⚠️ 'model' is missing. Using default 'model={model}'.")
    from ultralytics.yolo.engine.model import YOLO
    overrides['model'] = model
    model = YOLO(model, task='detect')
    if isinstance(overrides.get('pretrained'), str):
        model.load(overrides['pretrained'])

    # Mode
    if mode in ('predict', 'track') and 'source' not in overrides:
        overrides['source'] = DEFAULT_CFG.source or ROOT / 'assets' if (ROOT / 'assets').exists() \
            else 'https://ultralytics.com/images/bus.jpg'
        LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using default 'source={overrides['source']}'.")
    elif mode in ('train', 'val'):
        if 'data' not in overrides:
            overrides['data'] =  DEFAULT_CFG.data
            LOGGER.warning(f"WARNING ⚠️ 'data' is missing. Using default 'data={overrides['data']}'.")


    # Run command in python
    # getattr(model, mode)(**vars(get_cfg(overrides=overrides)))  # default args using default.yaml
    getattr(model, mode)(**overrides)  # default args from model

def check_cfg_mismatch(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Inputs:
        - custom (Dict): a dictionary of custom configuration options
        - base (Dict): a dictionary of base configuration options
    """
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    if mismatched:
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base)  # key list
            matches = [f'{k}={DEFAULT_CFG_DICT[k]}' if DEFAULT_CFG_DICT.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid kari-det argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e

def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings.
    The function considers cases where the first argument ends with '=' or the second starts with '=',
    as well as when the middle one is an equals sign.

    Args:
        args (List[str]): A list of strings where each element is an argument.

    Returns:
        List[str]: A list of strings where the arguments around isolated '=' are merged.
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == '=' and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f'={args[i + 1]}'
            del args[i + 1]
        elif arg.endswith('=') and i < len(args) - 1 and '=' not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f'{arg}{args[i + 1]}')
            del args[i + 1]
        elif arg.startswith('=') and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args

if __name__ == '__main__':
    # entrypoint(debug='kari-det train model=yolov8n.pt')
    entrypoint(debug='')