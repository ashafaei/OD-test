#!/bin/bash
source workspace/env/bin/activate
python -m visdom.server -port 8097 -env_path workspace/visdom 
