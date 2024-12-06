"""


"""


import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,

)
from transformers import  GPT2Config, GPT2LMHeadModel, GPT2Model

from typing import Optional, Tuple, Union, List

import os
import pickle
from transformers.utils import logging

from transformers import TrainerCallback, Trainer, TrainingArguments, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

import random
import math
import wandb
from omegaconf import DictConfig, OmegaConf


from synehrgy.config import HydraConfig

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



class SynEHRgy(GPT2LMHeadModel):
    def __init__(self, config):
        cfg = GPT2Config(
            vocab_size=config.total_vocab_size,
            n_positions=config.n_positions,
            n_ctx=config.n_ctx,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            eos_token_id=config.end_record_token_id,
            pad_token_id=config.pad_token_id,
            start_token_id=config.start_token_id,
            # end_label_token_id=config.end_label_token_id,
            # end_visit_token_id=config.end_visit_token_id,
            # anc_vocab_size=config.anc_vocab_size,
            # code_vocab_size=config.code_vocab_size,
            n_next=config.n_next,
            # strategy=config.strategy,
            # w_class=config.w_class,
            # emb_method=config.emb_method,
        )
        super().__init__(cfg)
        self.config = cfg

        self.transformer = GPT2Model(cfg)

        # if config.use_pretrained:
        #     print("Loading GPT2 pretrained model")
        #     # config = transformers.GPT2Config.from_pretrained('gpt2')
        #     self.transformer.h = GPT2Model.from_pretrained("gpt2").h



        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # self.fc1 = nn.Linear(config.vocab_size, config.vocab_size, bias=False)
        # set padding token id


        # multi token prediction head
        self.heads = nn.ModuleList()
        for _ in range(cfg.n_next):
            self.heads.append(nn.Linear(cfg.n_embd, cfg.vocab_size))

        

        # for soft labels
        ppp = "/mlodata1/hokarami/HALO_Inpatient/continuous_variables/data/mimic3-bigicd"
        self.M_soft_labels = torch.tensor(pickle.load(open(f"{ppp}/metadata2.pkl", "rb"))['M_soft_labels'], dtype=torch.float32, requires_grad=False).to('cuda')

        self.soft_labels = config.soft_labels



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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        loss = None


        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)

        if self.config.n_next > 1: # for multi token prediction
            z = hidden_states.detach()
            z.requires_grad = True
            for i in range(self.config.n_next):
                logits = self.heads[i](z)
                if i == 0:
                    lm_logits = logits
                if labels is not None:
                    # move labels to correct device to enable model parallelism
                    labels = labels.to(logits.device)
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., : -(i + 1), :].contiguous()
                    shift_labels = labels[..., (i + 1) :].contiguous()
                    # Flatten the tokens

                    loss_h = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    if self.training:
                        loss_h.backward()
                    if i == 0:
                        loss = loss_h

            if self.training:
                hidden_states.backward(z.grad)
        else:
            lm_logits = self.lm_head(hidden_states)  # + self.fc1(input_visits)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)

                shift_labels = labels[..., 1:].contiguous()
                shift_logits = lm_logits[..., :-1, :].contiguous()

                # alternative loss
                if self.training and self.soft_labels:
                    mask = shift_labels.view(-1) != self.config.pad_token_id  # True where we want to keep, False where to ignore
                    mask = mask.float()  # Convert mask to float
                    target_soft = self.M_soft_labels[shift_labels.view(-1)] * mask[:, None]

                    loss =  nn.CrossEntropyLoss(reduction='sum')(shift_logits.view(-1, shift_logits.size(-1)), target_soft)/mask.sum()
                else:

                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    


    def fit(self,cfg, train_dataset, eval_dataset, run_name='TEST'):
        
        PATH_SAVE_MODEL = "saved_models"
        
        assert train_dataset.is_tokenized == True, "train_dataset must be tokenized"
        assert eval_dataset.is_tokenized == True, "eval_dataset must be tokenized"

        # save the config in yaml file
        OmegaConf.save(config=cfg, resolve=True, f=f"{PATH_SAVE_MODEL}/{run_name}_config.yaml")
   



        training_args = TrainingArguments(
            output_dir=f'{PATH_SAVE_MODEL}/{run_name}',
            overwrite_output_dir=True,
            num_train_epochs=cfg.train.epochs,
            per_device_train_batch_size=cfg.hparams.mini_batch,
            per_device_eval_batch_size=cfg.hparams.mini_batch,
            learning_rate=3e-4,
            gradient_accumulation_steps = int(cfg.hparams.batch_size / cfg.hparams.mini_batch),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=100,
            save_total_limit=1,
            metric_for_best_model="eval_loss",           # use eval loss to determine the best model
            greater_is_better=False,                     # lower eval loss is better
            load_best_model_at_end=True,                 # load the best model at the end of training

            # bf16=True,
            dataloader_num_workers  = 4,
            report_to="wandb",
            
            run_name="m3-",

            logging_dir="./logs",
            logging_steps=10,
            logging_first_step=False,
            
            # eval_steps=100,

            
            

        )
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                PerplexityLoggingCallback(),
                EarlyStoppingCallback(early_stopping_patience=cfg.train.patience),  # early stopping callback
                ],
        )
        

        trainer.train()

        pass



    def generate_synthetic_dataset(self,cfg):

        self.eval()

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
                stoken =  [self.config.start_token_id]
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

        
        with torch.no_grad():

            ehr = self.generate(
                input_ids = context,
                attention_mask=attention_mask,
                max_length=self.config.n_ctx,
                num_return_sequences=1,
                **generation_config,
                # pad_token_id=pad_token_id,
            )

        return ehr.cpu().detach().numpy()

    @staticmethod
    def load_model(config_path, model_path):
        
        config = OmegaConf.load(f"{config_path}")
        config = HydraConfig(config)

        instance = SynEHRgy(config)


        # loading the model
        if os.path.exists(f"{model_path}"):
            print("[info] Loading model from ", model_path)
            # find all checkpointns
            checkpoints = [
                f
                for f in os.listdir(f"{model_path}")
            
            ]

            print("[info] Checkpoints found: ", checkpoints[0])

            # model = Bart9Model(config).to(device)

            try:
                # Load the weights from the safetensors file
                from safetensors import safe_open
                checkpoint_path = f"{model_path}/{checkpoints[0]}/model.safetensors"
                with safe_open(checkpoint_path, framework="pt", device='cuda') as f:
                    state_dict = {k: f.get_tensor(k) for k in f.keys()}

                instance.load_state_dict(state_dict)
            except:
                print("use bin file for loading model")
                checkpoint_path = f"{model_path}/{checkpoints[0]}/pytorch_model.bin"
                state_dict = torch.load(checkpoint_path)
                instance.load_state_dict(state_dict)

        return instance

        pass



