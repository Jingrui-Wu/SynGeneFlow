





from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def flatten_args(settings):
    result=[]
    for name,val in settings.items():
        flag='--'+name.replace('_','-')
        if isinstance(val,bool):
            if val:
                result.append(flag)
        elif isinstance(val,list):
            result.append(flag)
            result.extend(map(str,val))
        elif val is not None:
            result.extend((flag,str(val)))
    return result


def make_command(config, overrides=None):
    args=dict(config.get('args',{}))
    args.update(overrides or {})
    return [sys.executable,'-m','syngeneflow.train',*flatten_args(args)]


def main(argv=None):
    p=argparse.ArgumentParser(description='Launch SynGeneFlow with a JSON config')
    p.add_argument('--config',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    config=json.loads(Path(args.config).read_text())
    cmd=make_command(config,{'input':args.input,'output':args.output})
    print(shlex.join(cmd),flush=True)
    if not args.dry_run:
        subprocess.run(cmd,check=True)


if __name__=='__main__':
    main()
