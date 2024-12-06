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

    
    config_path = f"./saved_models/{RUN_NAME}_config.yaml"
    model_path = f"./saved_models/{RUN_NAME}"
    syn_folder = "./data/synthetic"

    
    # loading the model
    model = SynEHRgy.load_model(config_path, model_path).to(device)

    config = HydraConfig(OmegaConf.load(f"{config_path}"))
    metadata = pickle.load(open(config.dataset_folder+"/metadata2.pkl", "rb"))


    # generate synthetic data
    synthetic_data_tokenized = model.generate_synthetic_dataset(gen_cfg)


    # create a ClinicalDataset object
    synthetic_dataset = ClinicalDataset(syn_folder, split='synthetic', data=synthetic_data_tokenized, metadata=metadata)


    # detokenize
    synthetic_dataset.detokenize()

    # save the synthetic dataset to syn_folder
    synthetic_dataset.save(name = RUN_NAME)

    

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
