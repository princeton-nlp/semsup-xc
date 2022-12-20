import yaml
from typing import Dict
import re


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def read_yaml_config(file : str, output_dir : str = "", extra_args = {}) -> Dict:
    data = yaml.load(open(file), Loader=loader)
    data = {**{k: v for d in data if isinstance(data[d], dict) for k, v in data[d].items()}, **{k:v for k,v in data.items() if not isinstance(v, dict)}}
    data['output_dir'] = output_dir if output_dir else data['EXP_NAME']
    for e,v in extra_args.items():
        data[e] = v
    print(f'Yaml Config is:\n{"-" * 80}\n{data}\n{"-" * 80}')
    return data
