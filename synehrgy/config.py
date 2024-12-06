"""
    config file for SynEHRgy
"""

import pickle



class HydraConfig(object):
    def __init__(self, cfg):

        self.dataset_folder = cfg.data.path
        self.data_type = cfg.data.type

        metadata = pickle.load(open(f"{cfg.data.path}/metadata2.pkl", "rb"))
        token2id = metadata['token2id']
        idToLabel = metadata['idToLabel']

        self.code_vocab_size = metadata['vocab_size']['codes']
        self.label_vocab_size = len(idToLabel)
        
        if cfg.data.type == "cont":
            self.categorical_lab_vocab_size = metadata['vocab_size']['lab_cat']
            self.continuous_lab_vocab_size = metadata['vocab_size']['lab_cont']
            
            self.lab_vocab_size = self.categorical_lab_vocab_size + self.continuous_lab_vocab_size
            
            self.gap_vocab_size = metadata['vocab_size']['gap']
            self.covars_vocab_size = metadata['vocab_size']['covars']
        else:
            self.categorical_lab_vocab_size = 0
            self.continuous_lab_vocab_size = 0
            self.lab_vocab_size = 0
            self.continuous_vocab_size = 0

        self.total_code_vocab_size = self.code_vocab_size + self.lab_vocab_size + self.gap_vocab_size + self.covars_vocab_size
        # special tokens
        self.start_token_id = token2id['<s>']
        self.end_label_token_id = token2id['</label>']
        self.end_adm_token_id = token2id['</adm>']
        self.end_record_token_id = token2id['</s>']
        self.mask_token_id = token2id['<pad>']
        self.pad_token_id = token2id['<pad>']
        self.bfcast_token_id = token2id['<forecast>']
        self.efcast_token_id = token2id['</forecast>']
        self.decoder_start_token_id = token2id['<forecast>']


        self.total_vocab_size = len(token2id)
        # hparams
        self.lr = cfg.hparams.lr
        self.batch_size = cfg.hparams.batch_size
        self.n_ctx = cfg.hparams.n_ctx
        self.n_positions = cfg.hparams.n_ctx*0+1024 #1024
        self.n_embd = cfg.hparams.n_embd
        self.n_layer = cfg.hparams.n_layer
        self.n_head = cfg.hparams.n_head

        # loss params
        self.n_next = cfg.loss.n_next


        # train params
        self.epoch = cfg.train.epochs
        self.ratio = cfg.train.ratio

        if hasattr(cfg.train, "gclip"):
            self.gclip = cfg.train.gclip
        else:
            self.gclip = 0




        self.preprocess = cfg.preprocess


        
        self.soft_labels = cfg.loss.soft_labels


        return
