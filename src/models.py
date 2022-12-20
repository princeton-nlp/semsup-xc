'''
Initial Code taken from SemSup Repository. 
'''



import torch
from torch import nn
import sys
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Import configs
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.configuration_bert import BertConfig
import numpy as np
# Loss functions
from torch.nn import BCEWithLogitsLoss

from typing import Optional, Union, Tuple, Dict, List

import itertools

MODEL_FOR_SEMANTIC_EMBEDDING = {
    "roberta": "RobertaForSemanticEmbedding",
    "bert": "BertForSemanticEmbedding",
}

MODEL_TO_CONFIG = {
    "roberta": RobertaConfig,
    "bert": BertConfig,
}


def getLabelModel(data_args, model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.label_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModel.from_pretrained(
        model_args.label_model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return model, tokenizer



class AutoModelForMultiLabelClassification:
    """
    Class for choosing the right model class automatically.
    Loosely based on AutoModel classes in HuggingFace.
    """
    
    @staticmethod
    def from_pretrained(*args, **kwargs):
        # Check what type of model it is
        for key in MODEL_TO_CONFIG.keys():
            if type(kwargs['config']) == MODEL_TO_CONFIG[key]:
                class_name = getattr(sys.modules[__name__], MODEL_FOR_SEMANTIC_EMBEDDING[key])
                return class_name.from_pretrained(*args, **kwargs)
        
        # If none of the models were chosen
        raise("This model type is not supported. Please choose one of {}".format(MODEL_FOR_SEMANTIC_EMBEDDING.keys()))

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import XLNetForSequenceClassification, XLNetTokenizer

class BertForSemanticEmbedding(nn.Module):


    def __init__(self, config):
        # super().__init__(config)
        super().__init__()

        self.config = config

        self.coil = config.coil
        if self.coil:
            assert config.arch_type == 2
            self.token_dim = config.token_dim

        try: # Try catch was added to handle the ongoing hyper search experiments.
            self.arch_type = config.arch_type  
        except:
            self.arch_type = 2
        

        try:
            self.colbert = config.colbert
        except:
            self.colbert = False

        if config.encoder_model_type == 'bert':
            # self.encoder = BertModel(config)
            if self.arch_type == 1:
                self.encoder = AutoModelForSequenceClassification.from_pretrained(
                        'bert-base-uncased', output_hidden_states = True)
            else:
                self.encoder = AutoModel.from_pretrained(
                    config.model_name_or_path
                )
            # self.encoder = AutoModelForSequenceClassification.from_pretrained(
            #         'bert-base-uncased', output_hidden_states = True).bert
        elif config.encoder_model_type == 'roberta':
            self.encoder = RobertaForSequenceClassification.from_pretrained(
                    'roberta-base', num_labels = config.num_labels, output_hidden_states = True)
        elif config.encoder_model_type == 'xlnet':
            self.encoder = XLNetForSequenceClassification.from_pretrained(
                    'xlnet-base-cased', num_labels = config.num_labels, output_hidden_states = True)


        print('Config is', config)
        
        if config.negative_sampling == 'none':
            if config.arch_type == 1:
                self.fc1 = nn.Linear(5 * config.hidden_size, 512 if config.semsup else config.num_labels)
            elif self.arch_type == 3:
                self.fc1 = nn.Linear(config.hidden_size, 256 if config.semsup else config.num_labels)

        if self.coil:
            self.tok_proj = nn.Linear(self.encoder.config.hidden_size, self.token_dim)


        self.dropout = nn.Dropout(0.1)
        self.candidates_topk = 10
        if config.negative_sampling != 'none':
            self.group_y = np.array([np.array([l for l in group]) for group in config.group_y])
        #np.load('datasets/EUR-Lex/label_group_lightxml_0.npy', allow_pickle=True)

        self.negative_sampling = config.negative_sampling

        self.min_positive_samples = 20

        self.semsup = config.semsup
        self.label_projection = None
        if self.semsup:# and config.hidden_size != config.label_hidden_size:
            if self.arch_type == 1:
                self.label_projection = nn.Linear(512, config.label_hidden_size, bias= False)
            elif self.arch_type == 2:
                self.label_projection = nn.Linear(self.encoder.config.hidden_size, config.label_hidden_size, bias= False)
            elif self.arch_type == 3:
                self.label_projection = nn.Linear(256, config.label_hidden_size, bias= False)


        # self.post_init()    
    def compute_tok_score_cart(self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        if not self.colbert:
            qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)  # Q * LQ * 1 * 1
            doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)  # 1 * 1 * D * LD
            exact_match = doc_input_ids == qry_input_ids  # Q * LQ * D * LD
            exact_match = exact_match.float()
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, self.token_dim),  # (Q * LQ) * d
            doc_reps.view(-1, self.token_dim).transpose(0, 1)  # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2])  # Q * LQ * D * LD
        if self.colbert:
            scores, _ =  scores_no_masking.max(dim=3) 
        else:
            scores, _ = (scores_no_masking * exact_match).max(dim=3)  # Q * LQ * D
        tok_scores = (scores * qry_attention_mask.reshape(-1, qry_attention_mask.shape[-1]).unsqueeze(2))[:, 1:].sum(1)
        return tok_scores

    def coil_eval_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        desc_input_ids = None,
        desc_attention_mask = None,
        lab_reps = None,
        label_embeddings = None,
        clustered_input_ids = None,
        clustered_desc_ids = None, 

    ):

        outputs_doc, logits = self.forward_input_encoder(input_ids, attention_mask, token_type_ids)

        doc_reps = self.tok_proj(outputs_doc.last_hidden_state)  # D * LD * d
        # lab_reps = self.tok_proj(outputs_lab.last_hidden_state @ self.label_projection.weight)  # Q * LQ * d

        if clustered_input_ids is None:
            tok_scores = self.compute_tok_score_cart(
                    doc_reps, input_ids,
                    lab_reps, desc_input_ids.reshape(-1, desc_input_ids.shape[-1]), desc_attention_mask
            )
        else:
            tok_scores = self.compute_tok_score_cart(
                    doc_reps, clustered_input_ids,
                    lab_reps, clustered_desc_ids.reshape(-1, clustered_desc_ids.shape[-1]), desc_attention_mask
            )

        logits = self.semsup_forward(logits, label_embeddings.reshape(desc_input_ids.shape[0], desc_input_ids.shape[1], -1).contiguous(), same_labels= True)
        
        new_tok_scores = torch.zeros(logits.shape, device = logits.device)
        for i in range(tok_scores.shape[1]):
            stride = tok_scores.shape[0]//tok_scores.shape[1]
            new_tok_scores[i] = tok_scores[i*stride: i*stride + stride ,i]
        logits += new_tok_scores.contiguous()
        return logits


    
    def coil_forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        desc_input_ids: Optional[List[int]] = None,
        desc_attention_mask: Optional[List[int]] = None,
        desc_inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        clustered_input_ids = None,
        clustered_desc_ids = None, 
        ignore_label_embeddings_and_out_lab = None,
     ):
        # print(desc_input_ids.shape, desc_attention_mask.shape, desc_inputs_embeds.shape)
        outputs_doc, logits = self.forward_input_encoder(input_ids, attention_mask, token_type_ids)
        if ignore_label_embeddings_and_out_lab is not None:
            outputs_lab, label_embeddings = outputs_lab, label_embeddings
        else:
            outputs_lab, label_embeddings, _, _ = self.forward_label_embeddings(None, None, desc_input_ids = desc_input_ids, desc_attention_mask = desc_attention_mask, return_hidden_states = True, desc_inputs_embeds = desc_inputs_embeds)


        doc_reps = self.tok_proj(outputs_doc.last_hidden_state)  # D * LD * d
        lab_reps = self.tok_proj(outputs_lab.last_hidden_state @ self.label_projection.weight)  # Q * LQ * d

        if clustered_input_ids is None:
            tok_scores = self.compute_tok_score_cart(
                    doc_reps, input_ids,
                    lab_reps, desc_input_ids.reshape(-1, desc_input_ids.shape[-1]), desc_attention_mask
            )
        else:
            tok_scores = self.compute_tok_score_cart(
                    doc_reps, clustered_input_ids,
                    lab_reps, clustered_desc_ids.reshape(-1, clustered_desc_ids.shape[-1]), desc_attention_mask
            )


        logits = self.semsup_forward(logits, label_embeddings.reshape(desc_input_ids.shape[0], desc_input_ids.shape[1], -1).contiguous(), same_labels= True)
        
        new_tok_scores = torch.zeros(logits.shape, device = logits.device)
        for i in range(tok_scores.shape[1]):
            stride = tok_scores.shape[0]//tok_scores.shape[1]
            new_tok_scores[i] = tok_scores[i*stride: i*stride + stride ,i]
        logits += new_tok_scores.contiguous()

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        if not return_dict: 
            output = (logits,) + outputs_doc[2:] + (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs_doc.hidden_states,
            attentions=outputs_doc.attentions,
        )

    def semsup_forward(self, input_embeddings, label_embeddings, num_candidates = -1, list_to_set_mapping = None, same_labels = False):
        '''
            If same_labels = True, directly apply matrix multiplication
            else: num_candidates must not be -1, list_to_set_mapping must not be None
        '''
        if same_labels:
            logits = torch.bmm(input_embeddings.unsqueeze(1), label_embeddings.transpose(2,1)).squeeze(1)
        else:
            # TODO: Can we optimize this? Perhaps torch.bmm?
            logits = torch.stack(
                # For each batch point, calculate corresponding product with label embeddings
                [
                    logit @ label_embeddings[list_to_set_mapping[i*num_candidates: (i+1) * num_candidates]].T for i,logit in enumerate(input_embeddings)   
                ]
            )

        return logits

    def forward_label_embeddings(self, all_candidate_labels, label_desc_ids, desc_input_ids = None, desc_attention_mask = None, desc_inputs_embeds = None, return_hidden_states = False):
        # Given the candidates, and corresponding 
        # description numbers of labels
        # Returns the embeddings for unique label descriptions 
        
        if desc_attention_mask is None:
            num_candidates = all_candidate_labels.shape[1]
            # Create a set to perform minimal number of operations on common labels
            label_desc_ids_list = list(zip(itertools.chain(*label_desc_ids.detach().cpu().tolist()), itertools.chain(*all_candidate_labels.detach().cpu().tolist())))
            print('Original Length: ', len(label_desc_ids_list))
            label_desc_ids_set = torch.tensor(list(set(label_desc_ids_list)))
            print('New Length: ', label_desc_ids_set.shape)

            m1 = {tuple(x):i for i, x in enumerate(label_desc_ids_set.tolist())}
            list_to_set_mapping = torch.tensor([m1[x] for x in label_desc_ids_list])
            descs = [
                    self.tokenizedDescriptions[self.config.id2label[desc_lb[1].item()]][desc_lb[0]] for desc_lb in label_desc_ids_set
                ] 
            label_input_ids = torch.cat([
                desc['input_ids'] for desc in descs
            ])

            label_attention_mask = torch.cat([
                desc['attention_mask'] for desc in descs
            ])

            label_token_type_ids = torch.cat([
                desc['token_type_ids'] for desc in descs
            ])
            label_input_ids = label_input_ids.to(label_desc_ids.device)
            label_attention_mask = label_attention_mask.to(label_desc_ids.device)
            label_token_type_ids = label_token_type_ids.to(label_desc_ids.device)
            label_embeddings = self.label_model(
                label_input_ids,
                attention_mask=label_attention_mask,
                token_type_ids=label_token_type_ids,
            ).pooler_output
        else:
            list_to_set_mapping = None
            num_candidates = None
            if desc_inputs_embeds is not None:
                outputs = self.label_model(
                    inputs_embeds = desc_inputs_embeds.reshape(desc_inputs_embeds.shape[0] * desc_inputs_embeds.shape[1],desc_inputs_embeds.shape[2], desc_inputs_embeds.shape[3]).contiguous(),
                    attention_mask=desc_attention_mask.reshape(-1, desc_input_ids.shape[-1]).contiguous(),
                )
            else:
                outputs = self.label_model(
                    desc_input_ids.reshape(-1, desc_input_ids.shape[-1]).contiguous(),
                    attention_mask=desc_attention_mask.reshape(-1, desc_input_ids.shape[-1]).contiguous(),
                )
            label_embeddings = outputs.pooler_output
        if self.label_projection is not None:
            if return_hidden_states:
                return outputs, label_embeddings @ self.label_projection.weight, list_to_set_mapping, num_candidates
            else:
                return label_embeddings @ self.label_projection.weight, list_to_set_mapping, num_candidates
        else:
            return label_embeddings, list_to_set_mapping, num_candidates

    def forward_input_encoder(self, input_ids, attention_mask, token_type_ids, ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True if self.arch_type == 1 else False,
        )

        # Currently, method specified in LightXML is used
        if self.arch_type in [2,3]:
            logits = outputs[1]
        elif self.arch_type == 1:
            logits = torch.cat([outputs.hidden_states[-i][:, 0] for i in range(1, 5+1)], dim=-1)
        
        if self.arch_type in [1,3]:
            logits = self.dropout(logits)

        # No Sampling
        if self.arch_type in [1,3]:
            logits = self.fc1(logits)
        return outputs, logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_labels: Optional[torch.Tensor] = None,
        all_candidate_labels: Optional[torch.Tensor] = None,
        label_desc_ids: Optional[List[int]] = None,
        desc_inputs_embeds : Optional[torch.Tensor] = None,
        desc_input_ids: Optional[List[int]] = None,
        desc_attention_mask: Optional[List[int]] = None,
        label_embeddings : Optional[torch.Tensor] = None,
        clustered_input_ids: Optional[torch.Tensor] = None,
        clustered_desc_ids: Optional[torch.Tensor] = None, 
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:


        if self.coil:
            return self.coil_forward(
                input_ids, 
                attention_mask,
                token_type_ids,
                labels,
                desc_input_ids,
                desc_attention_mask,
                desc_inputs_embeds,
                return_dict,
                clustered_input_ids,
                clustered_desc_ids,
            )

        # STEP 2: Forward pass through the input model
        
        outputs, logits = self.forward_input_encoder(input_ids, attention_mask, token_type_ids)

        if self.semsup:
            if desc_input_ids is None:
                all_candidate_labels = torch.arange(labels.shape[1]).repeat((labels.shape[0], 1))
                label_embeddings, list_to_set_mapping, num_candidates = self.forward_label_embeddings(all_candidate_labels, label_desc_ids)
                logits = self.semsup_forward(logits, label_embeddings, num_candidates, list_to_set_mapping)
            else:
                label_embeddings, _, _ = self.forward_label_embeddings(None, None, desc_input_ids = desc_input_ids, desc_attention_mask = desc_attention_mask, desc_inputs_embeds = desc_inputs_embeds)
                logits = self.semsup_forward(logits, label_embeddings.reshape(desc_input_ids.shape[0], desc_input_ids.shape[1], -1).contiguous(), same_labels= True)

        elif label_embeddings is not None:
            logits = self.semsup_forward(logits, label_embeddings.contiguous() @ self.label_projection.weight, same_labels= True)
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

 
        if not return_dict: 
            output = (logits,) + outputs[2:] + (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

