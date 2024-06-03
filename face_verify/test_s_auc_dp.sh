#!/bin/bash
python verification.py --Net-type ArcFace-r50 --List ./text/tutorial_DP_celeba.txt --save_List ./results/text/ArcFace-r50-tutorial_DP_celeba.txt
python verification.py --Net-type ArcFace-r100  --List ./text/tutorial_DP_celeba.txt  --save_List ./results/text/ArcFace-r100-tutorial_DP_celeba.txt
#python AUC.py --save_List ./results/text/ArcFace-r100-tutorial_age_id.txt
python verification.py --Net-type CosFace-r50 --List ./text/tutorial_DP_celeba.txt   --save_List ./results/text/CosFace-r50-tutorial_DP_celeba.txt
#python AUC.py --save_List ./results/text/CosFace-r50-tutorial_age_id.txt
python verification.py --Net-type CosFace-r100 --List ./text/tutorial_DP_celeba.txt   --save_List ./results/text/CosFace-r100-tutorial_DP_celeba.txt
python verification.py --Net-type VGGFace2 --List ./text/tutorial_DP_celeba.txt  --save_List ./results/text/VGGFace2-tutorial_DP_celeba.txt