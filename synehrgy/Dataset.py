

import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from torch.utils.data import Dataset






def tokenize_dataset(dataset_orig, config, truncate=True, split=False, ignore_ts=False, bin_type='quantile',ts_shuffle=False):

    if bin_type=='quantile':
        token2id = pickle.load(open(f"{config.dataset_folder}/metadata.pkl", "rb"))['token2id']
        var2id = pickle.load(open(f"{config.dataset_folder}/metadata.pkl", "rb"))['var2id']
    else:
        token2id = pickle.load(open(f"{config.dataset_folder}/metadata2.pkl", "rb"))['token2id']
        var2id = pickle.load(open(f"{config.dataset_folder}/metadata2.pkl", "rb"))['var2id']
    dataset = []
    tok_horizons = []
    # for orig_ehr in tqdm(dataset_orig, desc="Tokenizing Dataset"):
    n_truncated_tokens = 0
    for orig_ehr in tqdm(dataset_orig, desc="Tokenizing Dataset"):
        
        n_stays = len(orig_ehr["codes"])

        new_ehr = []
        temp_horizon = [-1,-1,-1,-1]
        # add start record token
        new_ehr.append(token2id['<s>'])  # Start Record
        

        
        for i in range(n_stays):
            adm_labels_phe = orig_ehr["labels_phe"][i]
            adm_labels_ihm = orig_ehr["labels_ihm"][i]
            adm_covars = orig_ehr["covars"][i]
            adm_codes = orig_ehr["codes"][i]
            adm_ts = orig_ehr["ts"][i]
            adm_horizon = orig_ehr["horizons"][i]

            # add covars
            for var_id,disc_val in zip(adm_covars[0],adm_covars[1]):
                new_ehr.append(token2id[('covar',var_id,disc_val)])

            new_ehr.append(token2id['</covar>'])

            # Add Labels
            # add ihm label
            new_ehr.append(
                token2id[('label','ihm',adm_labels_ihm)]
            )
            # config.preprocess.label_shuffle
            all_labels_phe = np.random.permutation(adm_labels_phe.nonzero()[0]) if config.preprocess.label_shuffle else adm_labels_phe.nonzero()[0]
            
            for l in all_labels_phe:
                new_ehr.append(token2id[('label','phe',l)])

            

            new_ehr.append(
                token2id['</label>']
            )  # End Labels
            


            # # Add Covariates
            # for c in new_covariates:
            #     new_ehr.append(c + config.code_vocab_size + config.label_vocab_size)
            

        
            # Add code tokens
            for c in adm_codes:
                new_ehr.append(token2id[('code',c)])
            new_ehr.append(token2id['</code>'])
            
            # add ts tokens
            if not ignore_ts:
                for kk,v in enumerate(adm_ts):
                    if len(v[0]) == 0:
                        continue
                    var_ids = v[0]
                    disc_vals = v[1]
                    if ts_shuffle:
                        # shuffle var_ids and disc_vals in the same way
                        var_ids,disc_vals = zip(*random.sample(list(zip(var_ids,disc_vals)),len(var_ids)))


                    for var_id,disc_val in zip(var_ids,disc_vals):
                        new_ehr.append(token2id[('ts',var_id,disc_val)])

                    # add time gap
                    new_ehr.append(token2id[('timestamp',var2id['Hours'],v[2][0])])


                    if i==0: # only first stay
                        for i_h, hor in enumerate(adm_horizon):
                            if kk == hor:
                                temp_horizon[i_h] = len(new_ehr)
                               

                new_ehr.append(token2id['</ts>'])




            # add end adm token
            new_ehr.append(
                token2id['</adm>']
            )

        # add end record token
        new_ehr.append(
            token2id['</s>']
        )  # End Record
        

        if truncate:
            n_truncated_tokens += max(0, len(new_ehr) - config.n_ctx)
            new_ehr = new_ehr[:config.n_ctx]

        if split:
            # split into multiple records of size n_ctx
            while len(new_ehr) > config.n_ctx:
                dataset.append(new_ehr[: config.n_ctx])
                new_ehr = new_ehr[config.n_ctx :]


        dataset.append(new_ehr)
        tok_horizons.append(temp_horizon)

    assert len(tok_horizons) == len(dataset)
    n_tokens = sum([len(x) for x in dataset])
    if truncate:
        print(f"Truncated {n_truncated_tokens} tokens")
    if split:
        print(f"Split into {len(dataset)} records. original: {len(dataset_orig)}")
    print(f"current tokens: {n_tokens}")
    print(f"truncated/current tokens: {n_truncated_tokens/n_tokens*100}")

    return dataset, tok_horizons



def detokenize(synthetic_ehrs, config,id2token, idToCode=None):
    

    no_ihm = 0 # number of patients without ihm label
    

    n_full = 0
    n_trunc = 0
    ehr_outputs = []

    for i in tqdm(range(len(synthetic_ehrs))):
        seq_tokens = [id2token[x] for x in synthetic_ehrs[i]]
        
        
        all_labels_phe = []
        all_labels_ihm = []

        all_codes=[]
        all_ts = []
        all_covars = []

        current_label_phe = np.zeros(25, dtype=int)
        current_label_ihm = 0
        current_code = []
        current_ts = []
        current_covar = []
        
        start_token = False
        temp_code = []

        covar_vars, covar_vals = [],[]
        ts_vars, ts_vals = [],[]
        temp_label_phe = []
        temp_label_ihm = 0
        
        last_token = False

        for token in seq_tokens:
            
            if token == '<s>':
                start_token = True
            elif isinstance(token, tuple):
                
                if token[0] == 'covar':
                    covar_vars.append(token[1])
                    covar_vals.append(token[2])

                elif token[0] == 'label':
                    if token[1]=='phe':
                        temp_label_phe.append(token[2])
                    elif token[1]=='ihm':
                        temp_label_ihm = token[2]

                elif token[0] == 'code':
                    temp_code.append(token[1])

                elif token[0] == 'ts':
                    ts_vars.append(token[1])
                    ts_vals.append(token[2])


                elif token[0] == 'timestamp':
                    
                    current_ts.append((
                        
                        ts_vars,
                        ts_vals,
                        [token[2]]
                    ))
                    ts_vars, ts_vals = [],[]
            
            elif token == '</covar>':
                current_covar = (covar_vars, covar_vals)
                covar_vars, covar_vals   = [],[]

            elif token == '</label>':
                # current_label_phe = np.zeros(25, dtype=int)
                # set the labels
                for l in temp_label_phe:
                    current_label_phe[l] = 1
                temp_label_phe = []
                
                # current_label_ihm = 0
                # # if len(temp_label_ihm)>0:
                current_label_ihm = temp_label_ihm
                temp_label_ihm =0
                
                
                # else:
                #     current_label_ihm = 0
                #     no_ihm +=1
                    # print('ihm label not found')
            elif token == '</code>':
                current_code = temp_code
                temp_code = []
            
            elif token == '</adm>':
                all_labels_phe.append(current_label_phe)
                all_labels_ihm.append(current_label_ihm)
                all_codes.append(current_code)
                all_ts.append(current_ts)
                all_covars.append(current_covar)

                current_label_phe = np.zeros(25, dtype=int)
                current_label_ihm = 0
                current_code = []
                current_ts = []
                current_covar = []
            
            elif token == '</s>':
                last_token = True
                n_full += 1
                break
            
        if not last_token:
            n_trunc += 1
            
            all_labels_phe.append(current_label_phe)
            all_labels_ihm.append(current_label_ihm)
            all_codes.append(current_code)
            all_ts.append(current_ts)
            all_covars.append(current_covar)

        
        ehr_outputs.append({
            'covars': all_covars,
            'codes': all_codes,
            'ts': all_ts,
            'labels_phe': all_labels_phe,
            'labels_ihm': all_labels_ihm
        })


    print(f"full: {n_full}, truncated: {n_trunc}")
    print(f"no ihm: {no_ihm} / {len(synthetic_ehrs)}")
    


    
    return ehr_outputs




def tokenize_dataset_raw(dataset_orig, config,tokenizer, truncate=True, split=False, ignore_ts=False, bin_type='quantile',ts_shuffle=False):


        
    # n_before = len(dataset_orig)
    # dataset_orig = [
    #     patient
    #     for patient in dataset_orig
    #     if sum([len(visit) for visit in patient["visits"]])
    #     + len(patient["visits"])
    #     + config.label_vocab_size
    #     + 3
    #     < config.n_ctx
    # ]
    # n_after = len(dataset_orig)
    # print(f"Removed {n_before - n_after} patients from dataset")
    if bin_type=='quantile':
        token2id = pickle.load(open(f"{config.dataset_folder}/metadata.pkl", "rb"))['token2id']
        var2id = pickle.load(open(f"{config.dataset_folder}/metadata.pkl", "rb"))['var2id']
    else:
        token2id = pickle.load(open(f"{config.dataset_folder}/metadata2.pkl", "rb"))['token2id']
        var2id = pickle.load(open(f"{config.dataset_folder}/metadata2.pkl", "rb"))['var2id']
    dataset = []
    tok_horizons = []
    translation_table = str.maketrans('', '', "[],()'")
    # for orig_ehr in tqdm(dataset_orig, desc="Tokenizing Dataset"):
    n_truncated_tokens = 0
    for orig_ehr in tqdm(dataset_orig, desc="Tokenizing Dataset"):
        
        n_stays = len(orig_ehr["codes"])

        new_ehr = []
        temp_horizon = [-1,-1,-1,-1]
        # add start record token
        new_ehr.append(['<s>'])  # Start Record
        

        
        for i in range(n_stays):
            adm_labels_phe = orig_ehr["labels_phe"][i]
            adm_labels_ihm = orig_ehr["labels_ihm"][i]
            adm_covars = orig_ehr["covars"][i]
            adm_codes = orig_ehr["codes"][i]
            adm_ts = orig_ehr["ts"][i]
            adm_horizon = orig_ehr["horizons"][i]

            # add covars
            for var_id,disc_val in zip(adm_covars[0],adm_covars[1]):
                new_ehr.append([('covar',var_id,disc_val)])

            new_ehr.append(['</covar>'])

            # Add Labels
            # add ihm label
            new_ehr.append(
                [('label','ihm',adm_labels_ihm)]
            )
            # config.preprocess.label_shuffle
            all_labels_phe = np.random.permutation(adm_labels_phe.nonzero()[0]) if config.preprocess.label_shuffle else adm_labels_phe.nonzero()[0]
            
            for l in all_labels_phe:
                new_ehr.append([('label','phe',l)])

            

            new_ehr.append(
                ['</label>']
            )  # End Labels
            


            # # Add Covariates
            # for c in new_covariates:
            #     new_ehr.append(c + config.code_vocab_size + config.label_vocab_size)
            

        
            # Add code tokens
            for c in adm_codes:
                new_ehr.append([('code',c)])
            new_ehr.append(['</code>'])
            
            # add ts tokens
            if not ignore_ts:
                for kk,v in enumerate(adm_ts):
                    if len(v[0]) == 0:
                        continue
                    var_ids = v[0]
                    disc_vals = v[1]
                    if ts_shuffle:
                        # shuffle var_ids and disc_vals in the same way
                        var_ids,disc_vals = zip(*random.sample(list(zip(var_ids,disc_vals)),len(var_ids)))


                    for var_id,disc_val in zip(var_ids,disc_vals):
                        new_ehr.append([('ts',var_id,disc_val)])

                    # add time gap
                    new_ehr.append([('timestamp',var2id['Hours'],v[2][0])])


                    if i==0: # only first stay
                        for i_h, hor in enumerate(adm_horizon):
                            if kk == hor:
                                temp_horizon[i_h] = len(new_ehr)
                               

                new_ehr.append(['</ts>'])




            # add end adm token
            new_ehr.append(
                ['</adm>']
            )

        # add end record token
        new_ehr.append(
            ['</s>']
        )  # End Record
        
        new_ehr = tokenizer.encode(str(new_ehr).translate(translation_table))

        if truncate:
            n_truncated_tokens += max(0, len(new_ehr) - config.n_ctx)
            new_ehr = new_ehr[:config.n_ctx]

        if split:
            # split into multiple records of size n_ctx
            while len(new_ehr) > config.n_ctx:
                dataset.append(new_ehr[: config.n_ctx])
                new_ehr = new_ehr[config.n_ctx :]

        dataset.append(new_ehr)
        tok_horizons.append(temp_horizon)

    assert len(tok_horizons) == len(dataset)
    n_tokens = sum([len(x) for x in dataset])
    if truncate:
        print(f"Truncated {n_truncated_tokens} tokens")
    if split:
        print(f"Split into {len(dataset)} records. original: {len(dataset_orig)}")
    print(f"current tokens: {n_tokens}")
    print(f"truncated/current tokens: {n_truncated_tokens/n_tokens*100}")

    return dataset, tok_horizons






def mask_tokens(input_ids, mask_token_id, mask_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    Args:
        input_ids: list of token ids
        mask_token_id: the id of the token used for masking
        mask_probability: probability of masking each token
    Returns:
        masked_input_ids: masked input ids
        labels: labels for masked language modeling
    """
    labels = input_ids.copy()
    masked_input_ids = input_ids.copy()

    for i in range(len(input_ids)):
        if random.random() < mask_probability:
            rand = random.random()
            if rand < 0.8:
                masked_input_ids[i] = mask_token_id  # Replace with mask token
            # elif rand < 0.9:
            #     masked_input_ids[i] = random.randint(0, len(input_ids) - 1)  # Replace with random token
            # else keep original token (10% probability)

    return masked_input_ids, labels

def pad_inputs(inputs, config):
    max_len = config.n_ctx
    # padded_inputs = []
    # for i in inputs:
    #     padded_inputs.append(i + [config.pad_token_id] * (max_len - len(i)))
    return inputs[:max_len] + [config.pad_token_id] * (max_len - len(inputs))


class ClinicalDataset(Dataset):
    def __init__(self, path_processed,  split='test', data=None, metadata=None):

        self.path = path_processed
        self.split = split

        if split=='synthetic':

            self.data = data
            
            self.is_synthetic = True
            self.is_tokenized = True
            self.is_discretized = True

            self.metadata = metadata
            
            self.n_ctx = len(self.data[0])

            
            print("[info] Loaded synthetic dataset. Please note that the dataset is already tokenized and discretized")

        elif split in ['train','val','test']:

            self.data = pickle.load(open(path_processed+f"/{split}Dataset.pkl", "rb"))
            
            self.metadata = pickle.load(open(path_processed+"/metadata2.pkl", "rb"))
            
            self.is_synthetic = False
            self.is_discretized = False
            self.is_tokenized = False


        else:
            raise ValueError("split must be one of ['train','val','test','synthetic']")

        self.mask_token_id = self.metadata['token2id']['<pad>']
        self.pad_token_id = self.metadata['token2id']['<pad>']
        self.mask_probability = 0

        # print some stats
        print(f"[info] Loaded {split} dataset with {len(self.data)} patients")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if (not self.is_discretized) or (not self.is_tokenized):
            return self.data[idx]
        
        
        
        
        input_ids = self.data[idx]        

        masked_input_ids, labels = self._mask_tokens(input_ids)
        padded_masked_input_ids = self._pad_inputs(masked_input_ids)
        labels = self._pad_inputs(labels)

        # Create attention mask: 1 for real tokens, 0 for padding tokens
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in padded_masked_input_ids]

        return {
            'input_ids': torch.tensor(padded_masked_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
    
    


    def _mask_tokens(self, input_ids):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Args:
            input_ids: list of token ids
            mask_token_id: the id of the token used for masking
            mask_probability: probability of masking each token
        Returns:
            masked_input_ids: masked input ids
            labels: labels for masked language modeling
        """
        labels = input_ids.copy()
        masked_input_ids = input_ids.copy()

        for i in range(len(input_ids)):
            if random.random() < self.mask_probability:
                rand = random.random()
                if rand < 0.8:
                    masked_input_ids[i] = self.mask_token_id  # Replace with mask token
                # elif rand < 0.9:
                #     masked_input_ids[i] = random.randint(0, len(input_ids) - 1)  # Replace with random token
                # else keep original token (10% probability)

        return masked_input_ids, labels

    def _pad_inputs(self, inputs):
        max_len = self.n_ctx
        # padded_inputs = []
        # for i in inputs:
        #     padded_inputs.append(i + [config.pad_token_id] * (max_len - len(i)))
        return inputs[:max_len] + [self.pad_token_id] * (max_len - len(inputs))


    
    def get_index(self,mapping, key, value):
        # this function returns the index of the value in the mapping[key]
        possible_values = mapping[key]
        for i in range(len(possible_values) - 1):
            if value <= possible_values[i + 1]:
                return i
        if value > possible_values[-1]:
            return len(possible_values) - 2
        print(f"{value} for {key} not in {possible_values}")
        return int(len(possible_values)-2)

    def add_ts_data(self,df_ts):
        # this function discretizes the time series data

        adm_ts = []

        prev_time = 0
        # FLAG_24 = False
        # horizon = 1
        for time, mes in df_ts.iterrows():
            mes = {k:v for k,v in mes.items() if not pd.isnull(v)}
            # print(time)
            # print(mes)
            new_labs = []
            new_values = []
            for var, val in mes.items():
                if self.isCategorical[var]:
                    new_labs.append(
                        self.var2id[var]
                    )  # var2id[var] is the index in [17]
                    try:
                        new_values.append(
                            self.possibleValues[var][str(val)]
                        )  # why always the negative value? added +1 for correction
                    except:
                        print(f"Error Categorical: {var} {val}")
                else: # continuous

                    try:
                        new_values.append(self.get_index(self.discretization, var, float(val)))
                        new_labs.append(self.var2id[var])
                    except:
                        # print(f"Error Cont: {var} {val}")
                        pass
            
            

            time_gap = self.get_index(self.discretization, "Hours", time-prev_time)

                
            prev_time = time
            if len(new_labs) == len(new_values) and len(new_labs)>0:
                adm_ts.append((new_labs, new_values, [time_gap]))  # v[0] is empty
            else:
                # print("Error: different length of new_labs and new_values")
                pass
        return adm_ts
    def discretize(self, redo=False, cache=True):

        # reading from metadata
        self.possibleValues = self.metadata['possibleValues']
        self.discretization = self.metadata['discretization']
        self.var2id = self.metadata['var2id']
        self.token2id = self.metadata['token2id']
        self.isCategorical = self.metadata['isCategorical']
        self.codeToId = self.metadata['codeToId']
        self.ts_info = self.metadata['ts_info']
        
        
        if not self.is_discretized and not redo:

            # check if the discretized data is already saved
            if os.path.exists(self.path + f"/{self.split}DiscDataset.pkl"):
                print('[info] Discretized data already exists. Loading...')
                self.data = pickle.load(open(self.path + f"/{self.split}DiscDataset.pkl", "rb"))

                self.is_discretized = True
                return
        HORIZONS = [12,24,36,48] # EXPERIMENTAL
        COVARS = ['Age','Gender']



        print("[info] Discretizing data...")
        disc_data = []
        for p in tqdm(self.data[:]):
            all_covars = []
            all_codes = []
            all_ts = []
            all_horizons = []
            for i_stay in range(len(p['hadm_id'])):
                new_code = []
                covariates = p['covariates'][i_stay]
                
                # def add_covars():
                    
                x = []
                y = []
                for var in COVARS:
                    x.append(
                        self.var2id[var]
                    )
                    if self.isCategorical[var]:
                        
                        
                        y.append(
                                self.possibleValues[var][(covariates[COVARS.index(var)])]
                            ) 
                    else:
                        y.append(self.get_index(self.discretization, var,  covariates[COVARS.index(var)]))
                        # x.append(get_index(discretization, var, covariates[COVARS.index(var)]))

                all_covars.append((
                    
                    x,
                    y,
                    

                ))
                
                
                

                codes = p['codes'][i_stay]
                
                new_code = [self.codeToId[code] for code in (codes[0] + codes[1])]
                # code_ids

                # new_code.append(code_ids)
                all_codes.append(new_code)



                df_ts = p['ts'][i_stay].set_index('Hours')
                

                hours = df_ts.index.tolist()


                list_horizons = []
                for horizon in HORIZONS:
                    # find the maximum hour that is smaller than horizon
                    try:
                        max_hour = max([x for x in hours if x<horizon])
                        max_hour_id = hours.index(max_hour)
                    except:
                        print(f"Error: {horizon} {hours}")
                        max_hour_id = -1
                    # max_hour,    hours.index(max_hour)
                    list_horizons.append(max_hour_id)
                all_horizons.append(list_horizons)
                
                
                adm_ts = self.add_ts_data(df_ts)

                if horizon == 1:
                    a=1
                
                all_ts.append(adm_ts)


                
            # new_visits
            disc_patient = {
                'covars': all_covars,
                'codes': all_codes,
                'ts': all_ts,
                'labels_phe': p['label_phe'],
                'labels_ihm': p['label_ihm'],
                'horizons': all_horizons
            }
            disc_data.append(disc_patient)

        self.data = disc_data
        self.is_discretized = True

        # save if cache is True
        if cache:
            pickle.dump(self.data, open(self.path + f"/{self.split}DiscDataset.pkl", "wb"))
        pass

    

    def tokenize(self,n_ctx=1024, label_shuffle=False, truncate=True, split=False, ignore_ts=False,ts_shuffle=False):

        assert self.is_discretized, "Data must be discretized first"

        # print tokenization info
        print(f"[info] Tokenization setting: n_ctx={n_ctx}, label_shuffle={label_shuffle}, truncate={truncate}, split={split}, ignore_ts={ignore_ts}, ts_shuffle={ts_shuffle}")

        self.n_ctx = n_ctx

        token2id = self.token2id
        var2id = self.var2id

        dataset = []
        tok_horizons = []
        # for orig_ehr in tqdm(dataset_orig, desc="Tokenizing Dataset"):
        n_truncated_tokens = 0
        for orig_ehr in tqdm(self.data, desc="Tokenizing Dataset"):
            
            n_stays = len(orig_ehr["codes"])

            new_ehr = []
            temp_horizon = [-1,-1,-1,-1]
            # add start record token
            new_ehr.append(token2id['<s>'])  # Start Record
            

            
            for i in range(n_stays):
                adm_labels_phe = orig_ehr["labels_phe"][i]
                adm_labels_ihm = orig_ehr["labels_ihm"][i]
                adm_covars = orig_ehr["covars"][i]
                adm_codes = orig_ehr["codes"][i]
                adm_ts = orig_ehr["ts"][i]
                adm_horizon = orig_ehr["horizons"][i]

                # add covars
                for var_id,disc_val in zip(adm_covars[0],adm_covars[1]):
                    new_ehr.append(token2id[('covar',var_id,disc_val)])

                new_ehr.append(token2id['</covar>'])

                # Add Labels
                # add ihm label
                new_ehr.append(
                    token2id[('label','ihm',adm_labels_ihm)]
                )
                # config.preprocess.label_shuffle
                all_labels_phe = np.random.permutation(adm_labels_phe.nonzero()[0]) if label_shuffle else adm_labels_phe.nonzero()[0]
                
                for l in all_labels_phe:
                    new_ehr.append(token2id[('label','phe',l)])

                

                new_ehr.append(
                    token2id['</label>']
                )  # End Labels
                


                # # Add Covariates
                # for c in new_covariates:
                #     new_ehr.append(c + config.code_vocab_size + config.label_vocab_size)
                

            
                # Add code tokens
                for c in adm_codes:
                    new_ehr.append(token2id[('code',c)])
                new_ehr.append(token2id['</code>'])
                
                # add ts tokens
                if not ignore_ts:
                    for kk,v in enumerate(adm_ts):
                        if len(v[0]) == 0:
                            continue
                        var_ids = v[0]
                        disc_vals = v[1]
                        if ts_shuffle:
                            # shuffle var_ids and disc_vals in the same way
                            var_ids,disc_vals = zip(*random.sample(list(zip(var_ids,disc_vals)),len(var_ids)))


                        for var_id,disc_val in zip(var_ids,disc_vals):
                            new_ehr.append(token2id[('ts',var_id,disc_val)])

                        # add time gap
                        new_ehr.append(token2id[('timestamp',var2id['Hours'],v[2][0])])


                        if i==0: # only first stay
                            for i_h, hor in enumerate(adm_horizon):
                                if kk == hor:
                                    temp_horizon[i_h] = len(new_ehr)
                                

                    new_ehr.append(token2id['</ts>'])




                # add end adm token
                new_ehr.append(
                    token2id['</adm>']
                )

            # add end record token
            new_ehr.append(
                token2id['</s>']
            )  # End Record
            

            if truncate:
                n_truncated_tokens += max(0, len(new_ehr) - n_ctx)
                new_ehr = new_ehr[:n_ctx]

            if split:
                # split into multiple records of size n_ctx
                while len(new_ehr) > n_ctx:
                    dataset.append(new_ehr[: n_ctx])
                    new_ehr = new_ehr[n_ctx :]


            dataset.append(new_ehr)
            tok_horizons.append(temp_horizon)

        assert len(tok_horizons) == len(dataset)
        
        n_tokens = sum([len(x) for x in dataset])
        print(f"[info] Dataset size: {n_tokens / 1e6:.2f}M tokens")
        
        if truncate:
            print(f"[info] Truncated {n_truncated_tokens / 1e6:.2f}M tokens")
        if split:
            print(f"Split into {len(dataset)} records. original: {len(self.data)}")


        # print(f"truncated/current tokens: {n_truncated_tokens/n_tokens*100}")

        self.is_tokenized = True


        self.data = dataset

        # return dataset, tok_horizons

    
    def detokenize(self):
        
        if not self.is_tokenized:
            print("Data is not tokenized")
            return
        id2token =  {v:k for k,v in self.metadata['token2id'].items()}
        no_ihm = 0 # number of patients without ihm label
        

        n_full = 0
        n_trunc = 0
        ehr_outputs = []

        for i in tqdm(range(len(self.data)), desc="Detokenizing"):
            seq_tokens = [id2token[x] for x in self.data[i]]
            
            
            all_labels_phe = []
            all_labels_ihm = []

            all_codes=[]
            all_ts = []
            all_covars = []

            current_label_phe = np.zeros(25, dtype=int)
            current_label_ihm = 0
            current_code = []
            current_ts = []
            current_covar = []
            
            start_token = False
            temp_code = []

            covar_vars, covar_vals = [],[]
            ts_vars, ts_vals = [],[]
            temp_label_phe = []
            temp_label_ihm = 0
            
            last_token = False

            for token in seq_tokens:
                
                if token == '<s>':
                    start_token = True
                elif isinstance(token, tuple):
                    
                    if token[0] == 'covar':
                        covar_vars.append(token[1])
                        covar_vals.append(token[2])

                    elif token[0] == 'label':
                        if token[1]=='phe':
                            temp_label_phe.append(token[2])
                        elif token[1]=='ihm':
                            temp_label_ihm = token[2]

                    elif token[0] == 'code':
                        temp_code.append(token[1])

                    elif token[0] == 'ts':
                        ts_vars.append(token[1])
                        ts_vals.append(token[2])


                    elif token[0] == 'timestamp':
                        
                        current_ts.append((
                            
                            ts_vars,
                            ts_vals,
                            [token[2]]
                        ))
                        ts_vars, ts_vals = [],[]
                
                elif token == '</covar>':
                    current_covar = (covar_vars, covar_vals)
                    covar_vars, covar_vals   = [],[]

                elif token == '</label>':
                    # current_label_phe = np.zeros(25, dtype=int)
                    # set the labels
                    for l in temp_label_phe:
                        current_label_phe[l] = 1
                    temp_label_phe = []
                    
                    # current_label_ihm = 0
                    # # if len(temp_label_ihm)>0:
                    current_label_ihm = temp_label_ihm
                    temp_label_ihm =0
                    
                    
                    # else:
                    #     current_label_ihm = 0
                    #     no_ihm +=1
                        # print('ihm label not found')
                elif token == '</code>':
                    current_code = temp_code
                    temp_code = []
                
                elif token == '</adm>':
                    all_labels_phe.append(current_label_phe)
                    all_labels_ihm.append(current_label_ihm)
                    all_codes.append(current_code)
                    all_ts.append(current_ts)
                    all_covars.append(current_covar)

                    current_label_phe = np.zeros(25, dtype=int)
                    current_label_ihm = 0
                    current_code = []
                    current_ts = []
                    current_covar = []
                
                elif token == '</s>':
                    last_token = True
                    n_full += 1
                    break
                
            if not last_token:
                n_trunc += 1
                
                all_labels_phe.append(current_label_phe)
                all_labels_ihm.append(current_label_ihm)
                all_codes.append(current_code)
                all_ts.append(current_ts)
                all_covars.append(current_covar)

            
            ehr_outputs.append({
                'covars': all_covars,
                'codes': all_codes,
                'ts': all_ts,
                'labels_phe': all_labels_phe,
                'labels_ihm': all_labels_ihm
            })


        print(f"full: {n_full}, truncated: {n_trunc}")
        print(f"no ihm: {no_ihm} / {len(self.data)}")
        

        self.is_tokenized = False
        self.data = ehr_outputs
        
        # return ehr_outputs


    def save(self, name):
        if self.is_synthetic:
            assert not self.is_tokenized, "Please detokenize the data first"
            pickle.dump(
                self.data,
                open(f"{self.path}/{name}Dataset.pkl", "wb"),
            )

            print(f"[info] Saved synthetic dataset to {self.path}/{name}Dataset.pkl")



class MyDataset(Dataset):
    def __init__(self, data, config, mask_probability=0, truncate=True, split=False, ignore_ts=False, bin_type='quantile'):

        self.data, self.data_horizons = tokenize_dataset(data, config, truncate=truncate,split=split, bin_type = bin_type, ignore_ts=ignore_ts)

        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.mask_probability = mask_probability
        self.config = config
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        input_ids = self.data[idx]        

        masked_input_ids, labels = mask_tokens(input_ids, self.mask_token_id, self.mask_probability)
        padded_masked_input_ids = pad_inputs(masked_input_ids, self.config)
        labels = pad_inputs(labels, self.config)
        # Create attention mask: 1 for real tokens, 0 for padding tokens
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in padded_masked_input_ids]

        return {
            'input_ids': torch.tensor(padded_masked_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class MyDatasetRaw(Dataset):
    def __init__(self, data, config,tokenizer, mask_probability=0, truncate=True, split=False, ignore_ts=False, bin_type='quantile'):



        self.data, self.data_horizons = tokenize_dataset_raw(data, config,tokenizer, truncate=truncate,split=split, bin_type = bin_type, ignore_ts=ignore_ts)

        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.mask_probability = mask_probability
        self.config = config
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        input_ids = self.data[idx]        
        # input_ids = tokenize_dataset([self.data[idx]], self.config)[0]

        masked_input_ids, labels = mask_tokens(input_ids, self.mask_token_id, self.mask_probability)
        padded_masked_input_ids = pad_inputs(masked_input_ids, self.config)
        labels = pad_inputs(labels, self.config)
        # Create attention mask: 1 for real tokens, 0 for padding tokens
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in padded_masked_input_ids]

        return {
            'input_ids': torch.tensor(padded_masked_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

