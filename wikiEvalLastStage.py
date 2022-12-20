import itertools
import logging
import os
import random
import sys
import time
import json
from tqdm import tqdm
import pickle
from dataclasses import dataclass, field
from typing import Optional
import shutil

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    TrainerCallback,
)
from transformers import BertForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from src import getTokenizedLabelDescriptions
from src import getLabelModel
from src import semsup_data_collator
from src import SemSupDataset, DeViSEDataset
from src import AutoModelForMultiLabelClassification
from src import applyLightXMLNegativeSampling
from src import multilabel_metrics
from src import task_to_keys, task_to_label_keys, dataset_to_numlabels
from src import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from src import dataset_classification_type
from src import BertForSemanticEmbedding
from src import read_yaml_config
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


import json
coil_cluster_map = json.load(open('bert_coil_map_dict_lemma255K_isotropic.json'))
coil_cluster_map = {int(k):int(v) for k,v in coil_cluster_map.items()}

def setup_config(model_args, data_args, num_labels):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
    )
    config.model_name_or_path = model_args.model_name_or_path
    config.problem_type = dataset_classification_type[data_args.task_name]
    config.negative_sampling = model_args.negative_sampling
    config.semsup = model_args.semsup
    config.coil = model_args.coil
    config.token_dim = model_args.token_dim
    if config.semsup:
        # TODO: Again take from config somehow
        config.label_hidden_size = 512
    config.label_names = 'labels'
    # TODO: Take from config
    config.cluster_labels_dim = 64
    config.encoder_model_type = model_args.encoder_model_type
    config.arch_type = model_args.arch_type
    config.devise = model_args.devise
    config.normalize_embeddings = False
    return config

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(read_yaml_config(os.path.abspath(sys.argv[1]), output_dir='tmp'))

    isGZSL = sys.argv[2] == 'True' or sys.argv[2] == 'true'

    data_files = {
        'test':  'datasets/Wiki1M/test.jsonl' if isGZSL else 'datasets/Wiki1M/test_unseen.jsonl',
    }

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    raw_datasets2 = load_dataset(
        "json",
        data_files=data_files,
    )

    label_key = task_to_label_keys['wiki1m']
    label_list = [x.strip() for x in open('datasets/Wiki1M/all_labels.txt')]
    num_labels = len(label_list)
    label_list.sort() # For consistency

    config = setup_config(model_args, data_args, num_labels)


    temp_label_id = {v: i for i, v in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )

    model = BertForSemanticEmbedding(config)

    print('Starting label model loading')
    if model_args.semsup:
        label_model, label_tokenizer = getLabelModel(data_args, model_args)
        print('Gettting Tokenized Descs')
        model.label_model = label_model
        model.label_tokenizer = label_tokenizer

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    padding = "max_length"


    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    label_to_id = model.config.label2id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
    id2label = model.config.id2label

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        try: del input['labels']
        except: ...
        return result

    print('Attatching transform')

    raw_datasets = raw_datasets.with_transform(
        preprocess_function,
    )
    if model_args.semsup and data_args.large_dset and os.path.exists(data_args.tokenized_descs_file):
        if data_args.tokenized_descs_file.endswith('npy'):
            class_descs_tokenized = np.load(data_args.tokenized_descs_file, allow_pickle=True)
    print('Descs Load Completed')

    seen_labels = []
    UNSEEN_LABELS_FILE = 'datasets/Wiki1M/unseen_labels.txt'

    if UNSEEN_LABELS_FILE is not None:
        for line in open(UNSEEN_LABELS_FILE).readlines():
            seen_labels.append(line.strip())
    else:
        seen_labels = None
    if model_args.semsup and sys.argv[4] == 'embed_labels':
        if isGZSL:
            raw_datasets['test'] = SemSupDataset(raw_datasets['test'], data_args, data_args.descriptions_file,
            label_to_id, id2label, label_tokenizer, return_desc_embeddings = True,
            add_label_name = model_args.add_label_name, class_descs_tokenized = class_descs_tokenized, isTrain = False)
        else:
            raw_datasets['test'] = SemSupDataset(raw_datasets['test'], data_args, data_args.label_description_file,
            label_to_id, id2label, label_tokenizer, return_desc_embeddings = True,
            seen_labels = seen_labels, add_label_name = data_args.add_label_name)

    import torch

    data_collator = default_data_collator
    torch.cuda.empty_cache()
    if training_args.seed == -1:
        training_args.seed = np.random.randint(100000)
    trainer = Trainer(
        model=model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
    )


    OUT_DIR = sys.argv[3]
    model.load_state_dict(torch.load(f'{OUT_DIR}/pytorch_model.bin'))
    model.eval()
    testloader = trainer.get_test_dataloader(raw_datasets['test'])
    print(len(testloader.dataset))
    embedLabels = sys.argv[4] == 'embed_labels'
    embedInstances = sys.argv[4] == 'embed_instances'

    def compute_tok_score_cart(doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)  # Q * LQ * 1 * 1
        doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)  # 1 * 1 * D * LD
        exact_match = doc_input_ids == qry_input_ids  # Q * LQ * D * LD
        exact_match = exact_match.float()
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, 16),  # (Q * LQ) * d
            doc_reps.view(-1, 16).transpose(0, 1)  # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2])  # Q * LQ * D * LD
        scores, _ = (scores_no_masking * exact_match).max(dim=3)  # Q * LQ * D
        tok_scores = (scores * qry_attention_mask.reshape(-1, qry_attention_mask.shape[-1]).unsqueeze(2))[:, 1:].sum(1)

        return tok_scores


    from typing import Optional
    def coil_fast_eval_forward(
        input_ids: Optional[torch.Tensor] = None,
        doc_reps = None,
        logits: Optional[torch.Tensor] = None,
        desc_input_ids = None,
        desc_attention_mask = None,
        lab_reps = None,
        label_embeddings = None
    ):
        tok_scores = compute_tok_score_cart(
                doc_reps, input_ids,
                lab_reps.reshape(-1, lab_reps.shape[-2], lab_reps.shape[-1]), desc_input_ids.reshape(-1, desc_input_ids.shape[-1]), desc_attention_mask
        )
        logits = (logits.transpose(0, 1) @ label_embeddings.transpose(1,2)).squeeze()
        new_tok_scores = torch.zeros(logits.shape, device = logits.device)
        for i in range(tok_scores.shape[1]):
            stride = tok_scores.shape[0]//tok_scores.shape[1]
            new_tok_scores[i] = tok_scores[i*stride: i*stride + stride ,i]
        return (logits + new_tok_scores).squeeze()


    if embedInstances:
        dt = raw_datasets['test'].with_transform(preprocess_function)
        testloader = DataLoader(dt, collate_fn = default_data_collator, batch_size = training_args.per_device_eval_batch_size)

        # Lets get all the input documents
        inp_data = []#dict()
        import h5py
        l_f = h5py.File('wiki_labels.h5','r')
        f = h5py.File(sys.argv[5],'w')
        f.create_dataset('logits', shape = (len(testloader.dataset), 1000))
        f.create_dataset('labels', shape = (len(testloader.dataset), 1000))

        shortlists = h5py.File('datasets/Wiki1M/test_shortlists_4K_1000_bak.h5','r')['data']

        if True:
            inp_desc_ids = torch.zeros(1285321, 128, dtype = torch.long)
            inp_desc_mask = torch.zeros(1285321, 128, dtype = torch.long)
            inp_lab_reps = torch.zeros(1285321, 128, 16, dtype = torch.float)
            inp_label_embeddings = torch.zeros(1285321, 768, dtype = torch.float)

            # Load whole h5py File in memory
            for i, k in tqdm(enumerate(l_f.keys())):
                inp_desc_ids[int(k)] = torch.tensor(l_f[k]['desc_input_ids']).squeeze()
                inp_desc_mask[int(k)] = torch.tensor(l_f[k]['desc_attention_mask']).squeeze()
                inp_lab_reps[int(k)] = torch.tensor(l_f[k]['lab_reps']).squeeze()
                inp_label_embeddings[int(k)] = torch.tensor(l_f[k]['label_embeddings']).squeeze()

        inp_desc_ids = inp_desc_ids.apply_(coil_cluster_map.get)
        inp_desc_ids = inp_desc_ids.cuda()
        inp_desc_mask = inp_desc_mask.cuda()
        inp_label_embeddings = inp_label_embeddings.cuda()
        inp_lab_reps = inp_lab_reps.cuda()


        with torch.no_grad():
            ii = 0

            for item in tqdm(testloader):
                BS = item['input_ids'].shape[0]
                # if ii > 100:
                #     break
                batch_item = dict()
                item['input_ids'] = item['input_ids'].apply_(coil_cluster_map.get)
                batch_item['input_ids'] = item['input_ids'].cuda()
                batch_item['attention_mask'] = item['attention_mask'].cuda()
                batch_item['token_type_ids'] = item['token_type_ids'].cuda()
                outputs_doc, logits = model.forward_input_encoder(**batch_item)
                doc_reps = model.tok_proj(outputs_doc.last_hidden_state)
                desc_attention_mask = torch.stack([inp_desc_mask[shortlists[x]] for x in range(ii * BS, ii * BS + BS)])
                desc_input_ids = torch.stack([inp_desc_ids[shortlists[x]] for x in range(ii * BS, ii * BS + BS)])
                lab_reps = torch.stack([inp_lab_reps[shortlists[x]] for x in range(ii * BS, ii * BS + BS)])
                label_embeddings = torch.stack([inp_label_embeddings[shortlists[x]] for x in range(ii * BS, ii * BS + BS)])

                input_ids = batch_item['input_ids']
                logits = logits.unsqueeze(0)
                all_desc_attention_mask = desc_attention_mask
                all_desc_input_ids = desc_input_ids
                all_lab_reps = lab_reps
                all_label_embeddings = label_embeddings

                logits = coil_fast_eval_forward(input_ids, doc_reps, logits, all_desc_input_ids, all_desc_attention_mask, all_lab_reps, all_label_embeddings).cpu()
                f['logits'][ii * BS : ii * BS + BS] = logits
                all_labels = raw_datasets2['test'][ii * BS : ii * BS + BS]['label']
                for k in range(BS):
                    als = set(all_labels[k])
                    f['labels'][ii * BS + k] = [1. if id2label[x] in als else 0. for x in shortlists[ii * BS + k]]
                ii+=1
        f.close()

print(sys.argv)
main()