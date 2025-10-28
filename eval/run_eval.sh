#!/bin/bash

cd /home/ubuntu/PE-Field
source /home/ubuntu/PE-Field/envs/pe_field/bin/activate

python /home/ubuntu/PE-Field/eval/run_eval.py 2>&1 | tee /home/ubuntu/PE-Field/eval/eval.log

