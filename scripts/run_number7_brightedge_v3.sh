#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp
python $(< experiments/number_7_dodo_2pi_stage1_joint_nonorm_stratified6143_brightedgev3_skipprop2_rgbpinvprior_unscale_fromscratch_v1/artifacts/command.txt)
