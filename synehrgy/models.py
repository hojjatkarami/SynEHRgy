"""


"""



import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,

)
from transformers import  GPT2Config, GPT2LMHeadModel, GPT2Model, Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

from typing import Optional, Tuple, Union, List

import os
import pickle
from transformers.utils import logging

from transformers import TrainerCallback, Trainer, TrainingArguments, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

import random
import math
import wandb
from omegaconf import DictConfig, OmegaConf


from synehrgy.tokenizer import EHRTokenizer

logger = logging.get_logger(__name__)






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



class GPT2ModelCustom(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        if "anc" in config.strategy:
            self.anc_emb = nn.Embedding(config.anc_vocab_size + 1, config.n_embd)
            self.fc_1374 = nn.Linear(2 * config.n_embd, config.n_embd)

            self.CodeIndex_to_groupIndex = pickle.load(
                open("CodeIndex_to_groupIndex.pkl", "rb")
            )

        if "input" in config.strategy:
            self.fc_input = nn.Linear(config.vocab_size, config.n_embd)
            self.fc_1374 = nn.Linear(2 * config.n_embd, config.n_embd)
            pass

        if config.emb_method == "glove":
            print("Loading Glove embeddings")
            from glove import Corpus, Glove

            # Load the model
            glove = Glove.load("glove.model")

            # # Example: Get the vector for a word ID
            # word_id = 95
            # vector = glove.word_vectors[word_id]

            # glove.word_vectors.shape
            # pretrained_weight = torch.tensor(glove.word_vectors, dtype=torch.float32)
            pretrained_weight = self.wte.weight.detach().cpu().numpy()
            pretrained_weight[list(glove.dictionary.keys())] = torch.tensor(
                glove.word_vectors
            )
            self.wte = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_weight, dtype=torch.float32), freeze=False
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(
                    dtype=self.dtype
                )  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        ################### Added code BEGIN ###################
        if "anc" in self.config.strategy:
            lookup = [
                (
                    self.CodeIndex_to_groupIndex[input_id.item()]
                    if (input_id < self.config.code_vocab_size)
                    else self.config.anc_vocab_size
                )
                for input_id in input_ids.flatten()
            ]
            anc_ids = torch.tensor(lookup).to(device).view(input_ids.size())
            anc_embeds = self.anc_emb(anc_ids)

            if self.config.strategy == "anc_add":
                hidden_states = hidden_states + anc_embeds
            elif self.config.strategy == "anc_concat":
                hidden_states = torch.cat([hidden_states, anc_embeds], dim=-1)
                hidden_states = self.fc_1374(hidden_states)

        if "input" in self.config.strategy:
            input_ids2 = torch.zeros_like(input_ids)

            input_ids2[
                (input_ids == self.config.start_token_id)
                + (input_ids == self.config.end_label_token_id)
                + (input_ids == self.config.end_visit_token_id)
            ] = 1

            input_ids2 = torch.cumsum(input_ids2, 1)

            input_visits = torch.nn.functional.one_hot(
                input_ids, num_classes=self.config.vocab_size
            ).to(dtype=self.fc_input.weight.dtype)

            input_visits = torch.cumsum(input_visits, 1)

            # for input_id2, input_visit in zip(input_ids2, input_visits):
            #     for i in range(1, input_id2.max() + 1):
            #         input_visit[input_id2 == i] = input_visit[input_id2 == i].sum(0)[
            #             None, :
            #         ]

            if self.config.strategy == "input_add":
                hidden_states = hidden_states + self.fc_input(input_visits)
            elif self.config.strategy == "input_concat":
                hidden_states = torch.cat(
                    [hidden_states, self.fc_input(input_visits)], dim=-1
                )
                hidden_states = self.fc_1374(hidden_states)

        ################### Added cod END ###################

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class SynEHRgy(Trainer):
    def __init__(self,
                  config_main,
                    train_dataset=None,
                      eval_dataset=None,
                      token_list=None,
                      tokenizer=None,
                        run_name=None,
                          model=None):
        # self.config = config
        self.config_main = config_main
        self.context_length = config_main.n_ctx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = run_name


        import pickle
        metadata = pickle.load(open(f"{config_main.data.path}/metadata_{config_main.disc_name}.pkl", "rb"))
        idToLabel = metadata['idToLabel']

        

        # self.token2id = token2id

        if tokenizer is None:
            self.processing_class = EHRTokenizer(vocab_list=token_list)
        else:
            self.processing_class = tokenizer

        # tokenizer configs
        tokenizer_config = {
            "vocab_size": len(self.processing_class),
            "eos_token_id": self.processing_class.get_vocab()['</s>'],
            "pad_token_id": self.processing_class.get_vocab()['<pad>'],
            "bos_token_id": self.processing_class.get_vocab()['<s>'],
        }


        


        if model is None:
            model_class = config_main.model_config.config_class
            model_kwargs = {
                **config_main.model_config.config,
                **tokenizer_config,
            }

            if model_class == "GPT2Config":
                # Initialize GPT-2 model
                model_kwargs['n_positions'] = config_main.n_ctx
                cfg = GPT2Config(**model_kwargs)
                model = GPT2LMHeadModel(cfg).to(self.device)

            elif model_class == "Qwen2Config":
                # Initialize Qwen2 model
                cfg = Qwen2Config(**model_kwargs)
                model = Qwen2ForCausalLM(cfg).to(self.device)


        
        model.resize_token_embeddings(len(self.processing_class))

        # log model size in millions of parameters
        print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")

        if train_dataset is None:
            self.model = model
            return

        PATH_SAVE_MODEL = f"saved_models/{run_name}"
        
        

        os.makedirs(PATH_SAVE_MODEL, exist_ok=True)
        # save to wandb
        if wandb.run is not None:
            # if the project_name is eval
            if wandb.run.project == "SynEHRgy":
            
                wandb.log({"model_size": sum(p.numel() for p in model.parameters()) / 1_000_000})
                # save the config in pkl file
                import pickle
                # with open(f"{PATH_SAVE_MODEL}/config.pkl", "wb") as f:
                #     pickle.dump(config, f)
            
            
                # save config_main
                OmegaConf.save(config=config_main, resolve=True, f=f"{PATH_SAVE_MODEL}/config_main.yaml")

                # wandb.run.save(f"{PATH_SAVE_MODEL}/config.yaml")
                wandb.run.save(f"{PATH_SAVE_MODEL}/config_main.yaml")

                # log len(tokenizer)
                wandb.log({"tokenizer_length": len(self.processing_class.get_vocab())})
        
        


        
        if config_main.get("budget", None) is not None:
            # Compute number of training steps from budget (number of tokens)
            logger.info(f"Computing training steps from budget {config_main.budget:,} TFLOPs")

            def compute_training_flops(model_config, L: int) -> float:
                if model_config.config_class == "GPT2Config":
                    n_layer = model_config.config.n_layer
                    d = model_config.config.n_embd
                    n_head = model_config.config.n_head
                    forward_flops = n_layer * (12 * L * d**2 + 2 * L**2 * d)



                return forward_flops * 3 / 1e12

            tflop_per_example = compute_training_flops(
                config_main.model_config, L=self.context_length
            )
            logger.info(
                f"\tEstimated Training FLOPs per example: {tflop_per_example * 1e12:,.0f}"
            )

            tflop_per_step = (
                tflop_per_example
                * config_main.model_config.training.per_device_train_batch_size
                * config_main.model_config.training.gradient_accumulation_steps
            )
            self.tflop_per_step = tflop_per_step
            logger.info(f"\tEstimated Training TFLOPs per step: {tflop_per_step:.2f}")

            required_steps = int(config_main.budget / tflop_per_step)

            # update cfg.training.max_steps
            max_steps = required_steps
            logger.info(f"\tUpdated training.max_steps to {max_steps}")
        else:
            max_steps = -1


        bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8  # Ampere = 8+

        if not bf16_supported:
            config_main.model_config.training.bf16 = False
            config_main.model_config.training.fp16 = True
            print("BF16 not supported, using FP16 instead.")



        training_args = TrainingArguments(
            output_dir=f'{PATH_SAVE_MODEL}',
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=100,
            save_total_limit=1,
            metric_for_best_model="eval_loss",           # use eval loss to determine the best model
            greater_is_better=False,                     # lower eval loss is better
            load_best_model_at_end=True,                 # load the best model at the end of training

            num_train_epochs=config_main.train.num_train_epochs,
            max_steps=max_steps,


            
            **config_main.model_config.training,

            **config_main.training,
            # bf16=True,
            report_to="wandb",            
            # run_name="m3-",
            logging_dir="./logs",
            logging_steps=10,
            logging_first_step=False,
            
            # eval_steps=100,
        )

        if self.config_main.collate_fn == 'truncate':
            data_collator = self.collate_fn_truncate
        elif self.config_main.collate_fn == 'dense_packed':
            data_collator = self.collate_fn_dense_packed
        else:
            raise ValueError(f"Unknown collate_fn: {self.config_main.collate_fn}")
        print(f"Using data collator: {data_collator}")

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,

            callbacks=[
                PerplexityLoggingCallback(),
                EarlyStoppingCallback(early_stopping_patience=config_main.train.patience),  # early stopping callback
                ],

            data_collator=self.collate_fn_truncate,
            # compute_metrics=self._compute_metrics,

            processing_class=self.processing_class,
            # compute_loss_func=self.compute_loss_func,
        )

    def compute_embeddings(self, dataset, layer='last_hidden_state', batch_size=None):
        """
        Compute embeddings from the model's last hidden layer for all elements in the dataset.
        Returns a torch.Tensor (num_examples, hidden_dim).
        """
        self.model.eval()
        device = self.device

        # Reuse eval dataloader to ensure same collate/tokenization logic
        dataloader = self.get_eval_dataloader(dataset)

        all_embeddings = []

        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # move to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # forward pass
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True)

            # typically outputs.hidden_states is a tuple of [layer_0, layer_1, ..., layer_n]
            hidden = outputs.hidden_states[-1]  # last hidden layer

            # If using a language model: get embedding of last token, mean, or CLS token
            if "attention_mask" in batch:
                mask = batch["attention_mask"].unsqueeze(-1)
                emb = (hidden * mask).sum(1) / mask.sum(1)  # mean pooling over valid tokens
            else:
                emb = hidden.mean(1)

            all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def collate_fn_truncate(self, batch):
        # print(batch[0])
        tokenized = self.processing_class(
            batch,
            padding=True,
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.context_length,
        )

        # Create labels and mask out padding
        labels = tokenized["input_ids"].clone()
        labels[labels == self.processing_class.pad_token_id] = -100
        tokenized["labels"] = labels

        # print(tokenized["input_ids"][0])

        # log percentage that contains </s>
        # Compute fraction of eos tokens
        end_of_sequence = (tokenized["input_ids"] == self.processing_class.eos_token_id).sum(dim=1)
        
        end_of_sequence = end_of_sequence.sum().float() / end_of_sequence.numel()
        # print({"train/eos_ratio": end_of_sequence})

        assert tokenized["input_ids"].max() < len(self.processing_class), "Token id exceeds vocabulary size"
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
            # "types_disc": tokenized["types_disc"],
        }

    def collate_fn_dense_packed(self, batch):
        # Flatten all patient sequences into one long sequence
        packed_tokens = []
        for item in batch:
            packed_tokens.extend(
                item
            )  # already includes <BOS> and <EOS>

        # Optionally split into chunks of context_length
        chunks = [
            packed_tokens[i : i + self.context_length]
            for i in range(0, len(packed_tokens), self.context_length)
        ]

        tokenized = self.processing_class(
            chunks,
            padding=True,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )

         # Create labels and mask out padding
        labels = tokenized["input_ids"].clone()
        labels[labels == self.processing_class.pad_token_id] = -100
        tokenized["labels"] = labels

        # print(tokenized["input_ids"][0])

        # log percentage that contains </s>
        # Compute fraction of eos tokens
        end_of_sequence = (tokenized["input_ids"] == self.processing_class.eos_token_id).sum(dim=1)
        
        end_of_sequence = end_of_sequence.sum().float() / end_of_sequence.numel()
        # print({"train/eos_ratio": end_of_sequence})

        assert tokenized["input_ids"].max() < len(self.processing_class), "Token id exceeds vocabulary size"
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
            # "types_disc": tokenized["types_disc"],
        }
    
    def generate_synthetic_dataset(self,cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        # token2id = self.token2id
        synthetic_ehr_dataset = []
        selected_ids=[]

        for _ in tqdm(range(0, int(cfg.n_samples / 1), cfg.batch_size)):

            # bs = min([cfg.n_samples - i, cfg.batch_size])
            if cfg.fix_covars:
               
                
                # create subset horizon
                random_selection = random.sample(list(enumerate(zip(train_tokenized.data, train_tokenized.data_horizons))),  cfg.batch_size)

                
                temp = [x[0] for x in random_selection]

                data_sample = [x[1][0] for x in random_selection]
                horizon_sample = [x[1][1] for x in random_selection]
                
                if np.max(horizon_sample) > config.n_ctx:
                    print('HORIZON TOO LARGE ', np.max(horizon_sample))
                    continue

                selected_ids.extend(temp)
                print("DEBUG",len(temp),len(selected_ids))
                context = [
                     torch.tensor(x[:ii])
                       for x,ii in zip(data_sample, horizon_sample)]

                # # # # # # # # right padding
                # context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=token2id['<pad>']).to(device)


                # # # # # # # # # left padding
                # Determine the maximum length of sequences
                max_len = max([x.size(0) for x in context])
                # Apply left-padding manually
                left_padded_context = [
                    torch.cat([torch.full((max_len - x.size(0),), token2id['<pad>']), x])  # Left pad the sequence
                    for x in context
                ]
                # Pad the batch (now with left-padding) and move to the appropriate device
                context = torch.stack(left_padded_context).to(device)



                attention_mask = attention_mask = (context != token2id['<pad>']).long()




            else: # unconditional generation
                stoken =  [self.processing_class._convert_token_to_id('<s>')]  # start token
                context = (
                    torch.tensor(stoken, device=self.device, dtype=torch.long)
                    .unsqueeze(0)
                    .repeat( cfg.batch_size, 1)
                )
                attention_mask = None
            

            # Generate synthetic EHRs
            batch_synthetic_ehrs = self._sample_sequence(
                
                cfg.generation,
                context = context,
                attention_mask = attention_mask,
                # batch_size= cfg.batch_size,
                # device=device,
                # sample=True,
                # pad_token_id=token2id['<pad>'],                
            ) # (batch_size, n_ctx)

            # id2token =  {v:k for k,v in token2id.items()}
            # seq_tokens = [id2token[x] for x in batch_synthetic_ehrs[0]]
            # print(seq_tokens)
            # print('context', context[0], [id2token[x.item()] for x in context[0]])
            # term
            # batch_synthetic_ehrs = detokenize(batch_synthetic_ehrs, config,id2token)
            

            synthetic_ehr_dataset.extend( batch_synthetic_ehrs)

            # print(f"Generated {len(synthetic_ehr_dataset)} patients")

            if len(synthetic_ehr_dataset) > cfg.n_samples:

                synthetic_ehr_dataset = synthetic_ehr_dataset[:cfg.n_samples]
                selected_ids = selected_ids[:cfg.n_samples]
                break

            print(f"[info] Generated {len(synthetic_ehr_dataset)} synthetic patients")

        return synthetic_ehr_dataset


    def _sample_sequence(
        self,
        generation_config,
        context,
        attention_mask=None,
        # batch_size=None,
        # device="cuda",
        # sample=True,
        # pad_token_id=5127,
    ):

        assert self.model.config.eos_token_id == self.processing_class.eos_token_id
        
        with torch.no_grad():

            print('lets generate',context.shape, self.context_length, self.model.device)
            output = self.model.generate(
                input_ids = context,
                attention_mask=attention_mask,
                max_length=self.context_length,
                num_return_sequences=1,
                **generation_config,
                # pad_token_id=pad_token_id,
                use_cache=True,
                return_dict_in_generate=True
            )
            # last_hidden_state = output.last_hidden_state
            ehr = output.sequences
            print('done generate')

        return ehr.cpu().detach().numpy()

    @staticmethod
    def from_pretrained(model_path, train_dataset=None, eval_dataset=None, token_list=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        import pickle
        # config = pickle.load(open(f"{model_path}/config.pkl", "rb"))
        
        
        config_main = OmegaConf.load(f"{model_path}/config_main.yaml")

        from transformers import AutoModelForCausalLM

        
        base_path = model_path  # your main directory
        checkpoints = [d for d in os.listdir(base_path) if d.startswith("checkpoint-")]
        if not checkpoints:
            last_checkpoint = base_path
        else:
            # Sort by step number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            last_checkpoint = os.path.join(base_path, checkpoints[-1])

        print(f"Loading from: {last_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint).to(device)

        # load tokenizer
        tokenizer = EHRTokenizer.from_pretrained(last_checkpoint)
        
        return SynEHRgy(config_main,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        token_list=token_list,
                        tokenizer=tokenizer,
                        run_name="IGNORE",
                        model=model
                        )
        
