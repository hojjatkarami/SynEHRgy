import os
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
import math



from safetensors.torch import load_file
from transformers import TrainerCallback, Trainer, TrainingArguments, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import  GPT2LMHeadModel, GPT2Tokenizer



import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import wandb


from synehrgy.Dataset import MyDataset,MyDatasetRaw,ClinicalDataset
from synehrgy.utils import *
from synehrgy.config import HydraConfig
from synehrgy.models import SynEHRgy



PATH_SAVE_MODEL = "saved_models"




class PerplexityLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                perplexity = math.exp(logs['loss'])
                logs['perplexity'] = perplexity
                wandb.log({'train/perplexity': perplexity})

            if 'eval_loss' in logs:
                eval_perplexity = math.exp(logs['eval_loss'])
                logs['eval_perplexity'] = eval_perplexity
                wandb.log({'eval/perplexity': eval_perplexity})

@hydra.main(config_path="configs", config_name="configTrain", version_base=None)
def main(cfg: DictConfig):



    RUN_NAME = cfg.run_name
    
    # config = HydraConfig(cfg, tok_strategy=cfg.tok_strategy)
    
    # Data Loading

        # load processed data
    train_dataset = ClinicalDataset(cfg.data.path, cfg.n_ctx, load_discretized=True,split='train', disc_name=cfg.disc_name, tok_strategy=cfg.tok_strategy)
    eval_dataset = ClinicalDataset(cfg.data.path, cfg.n_ctx, load_discretized=True,split='val', disc_name=cfg.disc_name, tok_strategy=cfg.tok_strategy)

        # discretize the data
    # train_dataset.discretize(tok_strategy=cfg.tok_strategy, disc_name=cfg.disc_name)
    # eval_dataset.discretize(tok_strategy=cfg.tok_strategy, disc_name=cfg.disc_name)

    token_list = train_dataset.get_token_list(tok_strategy=cfg.tok_strategy)

    # tokenize the data
    # train_dataset.tokenize(n_ctx=cfg.n_ctx, tok_strategy=cfg.tok_strategy)
    # eval_dataset.tokenize(n_ctx=cfg.n_ctx, tok_strategy=cfg.tok_strategy)


    
    # Example usage:
    wandb.init(project=cfg.wandb.project, name=RUN_NAME, config=OmegaConf.to_container(cfg, resolve=True))
    # log_flat(run, cfg)

    # build the model
    trainer = SynEHRgy(cfg,
                    #    config,
                     train_dataset=train_dataset,
                     eval_dataset=eval_dataset,
                     token_list=token_list,
                     run_name=RUN_NAME
                     )

    trainer.train()
    # # train the model
    # model.fit(cfg, train_dataset, eval_dataset, run_name=RUN_NAME)



if __name__ == "__main__":

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 4
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)



    main()
