import json
import pathlib
from copy import deepcopy


def get_settings(fpath: pathlib.Path = pathlib.Path('./settings.json')) -> dict:
    with open(fpath, 'r') as fp:
        jobj = json.load(fp)
        for path in jobj['paths']:
            if path in jobj:
                jobj[path] = pathlib.Path(jobj[path])
        return jobj


def set_settings(settings: dict, fpath: pathlib.Path = pathlib.Path('./settings.json')):
    with open(fpath, 'w+') as fp:
        jobj = deepcopy(settings)
        for path in settings['paths']:
            if path in jobj:
                jobj[path] = str(jobj[path])
        json.dump(jobj, fp, indent=2)
