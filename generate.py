import torch
import pickle
import random
import numpy as np
from tqdm import tqdm


import torch.nn.functional as F
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# our modules
from synehrgy.models import SynEHRgy
from synehrgy.config import HydraConfig
from synehrgy.Dataset import MyDataset, detokenize, ClinicalDataset



PATH_SAVE_MODEL = "saved_models"
PATH_GEN = "data/synthetic"

def sample_sequence(
    model,
    length,
    generation_config,
    context,
    attention_mask=None,
    # batch_size=None,
    # device="cuda",
    # sample=True,
    # pad_token_id=5127,
):

    
    with torch.no_grad():

        ehr = model.generate(
            input_ids = context,
            attention_mask=attention_mask,
            max_length=length,
            num_return_sequences=1,
            **generation_config,
            # pad_token_id=pad_token_id,
        )

    return ehr.cpu().detach().numpy()



@hydra.main(config_path="configs", config_name="configGenerate", version_base=None)
def main(gen_cfg: DictConfig):

    RUN_NAME = gen_cfg.run_name

    
    # config_path = f"./saved_models/{RUN_NAME}_config.yaml"
    model_path = f"./saved_models/{RUN_NAME}"
    syn_folder = "./data/synthetic"

    

    
    # loading the model
    trainer = SynEHRgy.from_pretrained(model_path)

    # config = pickle.load(open(f"{model_path}/config.pkl", "rb"))

    config_main = OmegaConf.load(f"{model_path}/config_main.yaml")
    # config.n_ctx = config_main.n_ctx
    # config.disc_name = config_main.disc_name

    # metadata = pickle.load(open(f"{config_main.data.path}/metadata_{config_main.disc_name}.pkl", "rb"))
    
    # if config_main.tok_strategy == "var+quant":
    #     token2id = metadata['token2id']

    #     # Separate tokens by type
    #     temp_non_ts = [k for k in token2id.keys() if k[0] != 'ts']
    #     temp_ts = [k for k in token2id.keys() if k[0] == 'ts']

    #     # Extract time-series variable and quantile tokens
    #     temp_var = sorted({('ts', k[1]) for k in temp_ts if len(k) > 1})
    #     temp_quant = sorted({('quant', k[2]) for k in temp_ts if len(k) > 2})

    #     # Combine in deterministic order
    #     token2id_new = {}
    #     for k in temp_non_ts:
    #         token2id_new[k] = len(token2id_new)
    #     for k in temp_var:
    #         token2id_new[k] = len(token2id_new)
    #     for k in temp_quant:
    #         token2id_new[k] = len(token2id_new)

    #     token2id = token2id_new
    #     metadata['token2id'] = token2id


    # temp_timestamp = [k for k in token2id.keys() if k[0] == 'timestamp']
    # print('# timestamp tokens:', (temp_timestamp))
    # term

    # generate synthetic data
    synthetic_data_tokenized = trainer.generate_synthetic_dataset(gen_cfg)


    # create a ClinicalDataset object
    synthetic_dataset = ClinicalDataset(config_main.data.path, config_main.n_ctx, split='synthetic', data=synthetic_data_tokenized, disc_name=config_main.disc_name)


    # detokenize
    synthetic_dataset.detokenize(trainer.processing_class)

    # save the synthetic dataset to syn_folder
    synthetic_dataset.save(syn_folder, run_name=RUN_NAME)

    

    return


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    main()
