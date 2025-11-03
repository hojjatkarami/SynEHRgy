#!/bin/bash

source activate paper2025
cd /home/hokarami/code/SynEHRgy

# v2 old publised

# v3 new trainer
# python train.py run_name=synehrgy-mimic-v33 model_config=gpt2
# python generate.py run_name=synehrgy-mimic-v3

# source activate synehrgy_results
# python generate_results.py data_name=synehrgy-mimic-v3

# qwenn-small
# python train.py run_name=mimic3-qwen-s model_config=qwen-small
# python generate.py run_name=mimic3-qwen-s
# python generate_results.py data_name=mimic3-qwen-s



# v4 new trainer with quantile binning
# python train.py run_name=v6-gpt3 model_config=gpt2 disc_name=quantile_v1
# python generate.py run_name=eq-quant-v4 disc_name=quantile_v1
# python generate_results.py data_name=eq-quant-v4

# # v5 new trainer with quantile binning
# python train.py run_name=ueq-quant-v5 model_config=gpt2 disc_name=quantile_ueq_v1
# python generate.py run_name=ueq-quant-v5 
# python generate_results.py data_name=ueq-quant-v5


# v6 gpt3 : gpt2 with additional trainier params
# python train.py run_name=v6-gpt3 model_config=gpt3 disc_name=uniform_v1
# python generate.py run_name=v6-gpt3 disc_name=uniform_v1
# python generate_results.py data_name=v6-gpt3


# v7 gpt3 + new tokenization var+quantile instead of var_quantile
# python train.py run_name=v7-gpt3-var+quant model_config=gpt3 disc_name=uniform_v1 tok_strategy=var+quant
# python generate.py run_name=v7-gpt3-var+quant #disc_name=uniform_v1 tok_strategy=var+quant
# python generate_results.py data_name=v7-gpt3-var+quant


# v8 gpt3 + new tokenization var+quantile instead of var_quantile
# python train.py run_name=v8-gpt3-var+quant model_config=gpt3 disc_name=uniform_v1 tok_strategy=var+quant
# python generate.py run_name=v8-gpt3-var+quant #disc_name=uniform_v1 tok_strategy=var+quant
# python generate_results2.py data_name=v8-gpt3-var+quant
# python generate_results2.py data_name=v8-gpt3-var+quant var_only=true



# v9 gpt3 + new tokenization var+quantile instead of var_quantile
# python train.py run_name=v9-gpt3-var+quant model_config=gpt3 disc_name=uniform_v1 tok_strategy=var+quant n_ctx=1024
# python generate.py run_name=v9-gpt3-var+quant #disc_name=uniform_v1 tok_strategy=var+quant
# python generate_results2.py data_name=v9-gpt3-var+quant



# v10 gpt3 + new tokenization var+quantile instead of var_quantile
python train.py run_name=v10-dp-gpt3-var+quant model_config=gpt3 disc_name=uniform_v1 tok_strategy=var+quant collate_fn=dense_packed
python generate.py run_name=v10-dp-gpt3-var+quant 
# python generate_results2.py data_name=v10-dp-gpt3-var+quant
# python generate_results2.py data_name=v10-dp-gpt3-var+quant var_only=true







# v7 gpt3 + new tokenization var+quantile instead of var_quantile
# python train.py run_name=v7-eicu-gpt3-var+quant model_config=gpt3 disc_name=uniform_v1 tok_strategy=var+quant data=eicu
# python generate.py run_name=v7-eicu-gpt3-var+quant #disc_name=uniform_v1 tok_strategy=var+quant
# python generate_results.py data_name=v7-eicu-gpt3-var+quant



# configs
# n_ctx=1024,2048,4096
# tok_strategy=var+quant,var_quant
# disc_name=uniform_v1,quantile_v1
# data: mimic3,eicu
# collate: truncate,dense_packed