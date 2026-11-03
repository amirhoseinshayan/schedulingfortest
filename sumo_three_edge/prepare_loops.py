"""Backward-compatible CLI entry point. Fixed loops are NO LONGER used.

This shim preserves old shell calls while generating random SUMO rerouters.
"""
import argparse
import json
from prepare_rerouters import make

def prepare(seed=42,required_seconds=600.0):
    return make(seed,required_seconds)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--seconds',type=float,default=600.0)
    x=a.parse_args()
    print(json.dumps(prepare(x.seed,x.seconds),indent=2))
