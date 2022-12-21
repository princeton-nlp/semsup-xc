# USAGE: python fastInferenceStage1.py config_file.yml True    checkpoint_path embed_instances   out_file
#                                                 isGZSL           <or embed_labels>
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


DSET_NAME = sys.argv[6] # Wiki1M or Amzn13K 

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
        'test':  f'datasets/{DSET_NAME}/test.jsonl' if isGZSL else f'datasets/{DSET_NAME}/test_unseen_split6500_2.jsonl' if DSET_NAME == 'Amzn13K' else 'test_unseen.jsonl',
    }

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    # label_key = task_to_label_keys['eurlex57k']
    if DSET_NAME == 'Amzn13K':
        label_key = task_to_label_keys['amazon13k'] 
    else:
        label_key = task_to_label_keys['wiki1m'] 
    label_list = [x.strip() for x in open(f'datasets/{DSET_NAME}/all_labels.txt')]
    num_labels = len(label_list)
    label_list.sort() # For consistency

    config = setup_config(model_args, data_args, num_labels)


    temp_label_id = {v: i for i, v in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )

    model = BertForSemanticEmbedding(config)


    if model_args.semsup:
        label_model, label_tokenizer = getLabelModel(data_args, model_args)
        if data_args.large_dset:
            tokenizedDescriptions = getTokenizedLabelDescriptions(data_args, data_args.descriptions_file, label_tokenizer)
            model.tokenizedDescriptions = tokenizedDescriptions
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

        if label_to_id is not None and label_key in examples:
            if isinstance(examples[label_key][0], list):
                labels = [[label_to_id[l] for l in examples[label_key][i]] for i in range(len(examples[label_key]))]
                result["label"] = [[1 if j in labels[i] else 0 for j in range(num_labels)] for i in range(len(labels))]
            else:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

        
        try: del input['labels']
        except: ...

        return result

    raw_datasets = raw_datasets.with_transform(
        preprocess_function,
    )
    if model_args.semsup and data_args.large_dset and os.path.exists(data_args.tokenized_descs_file):
        if data_args.tokenized_descs_file.endswith('npy'):
            class_descs_tokenized = np.load(data_args.tokenized_descs_file, allow_pickle=True)
    print('Descs Load Completed')

    seen_labels = []
    UNSEEN_LABELS_FILE = f'datasets/{DSET_NAME}/unseen_labels_split6500_2.txt' if DSET_NAME == 'Amzn13K' else f'datasets/{DSET_NAME}/unseen_labels.txt'

    if UNSEEN_LABELS_FILE is not None:
        for line in open(UNSEEN_LABELS_FILE).readlines():
            seen_labels.append(line.strip())
    else:
        seen_labels = None
    if model_args.semsup and sys.argv[4] == 'embed_labels':
        if isGZSL:
            raw_datasets['test'] = SemSupDataset(raw_datasets['test'], data_args, data_args.descriptions_file,
            label_to_id, id2label, label_tokenizer, return_desc_embeddings = True,
            add_label_name = model_args.add_label_name, class_descs_tokenized = None if DSET_NAME == 'Amzn13K' else class_descs_tokenized, isTrain = False)
        else:
            raw_datasets['test'] = SemSupDataset(raw_datasets['test'], data_args, data_args.label_description_file,
            label_to_id, id2label, label_tokenizer, return_desc_embeddings = True,
            seen_labels = seen_labels, add_label_name = data_args.add_label_name)
        

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


    if embedLabels:
        if data_args.large_dset:
            testloader = trainer.get_test_dataloader(raw_datasets['test'])
            print(len(testloader.dataset))
            import h5py
            f = h5py.File(sys.argv[5],'w')
            with torch.no_grad():
                tokenized_descs = testloader.dataset.class_descs_tokenized
                label_data = dict()
                for ii, (inp_ids, atm) in tqdm(enumerate(zip(tokenized_descs['input_ids'], tokenized_descs['attention_mask']))):
                    for i in list(np.random.choice(inp_ids.shape[0], 1)): #range(len(tokenized_descs[label][0])):
                        desc_input_ids = torch.from_numpy(np.array(inp_ids[i])).unsqueeze(0).cuda()
                        desc_attention_mask = torch.from_numpy(np.array(atm[i])).unsqueeze(0).cuda()
                        outputs_lab, label_embeddings, _, _ = model.forward_label_embeddings(None, None, desc_input_ids = desc_input_ids, desc_attention_mask = desc_attention_mask, return_hidden_states = True)
                        lab_reps = model.tok_proj(outputs_lab.last_hidden_state @ model.label_projection.weight)
            f.close()
        else:
            testloader = trainer.get_test_dataloader(raw_datasets['test'])
            print(len(testloader.dataset))
            with torch.no_grad():
                tokenized_descs = testloader.dataset.class_descs_tokenized
                label_data = dict()
                for label in tqdm(tokenized_descs):
                    label_data[label] = []
                    for i in range(len(tokenized_descs[label][0])):
                        if i == 20:
                            break
                        desc_input_ids = torch.from_numpy(np.array(tokenized_descs[label][0][i])).unsqueeze(0).cuda()
                        desc_attention_mask = torch.from_numpy(np.array(tokenized_descs[label][1][i])).unsqueeze(0).cuda()
                        outputs_lab, label_embeddings, _, _ = model.forward_label_embeddings(None, None, desc_input_ids = desc_input_ids, desc_attention_mask = desc_attention_mask, return_hidden_states = True)
                        lab_reps = model.tok_proj(outputs_lab.last_hidden_state @ model.label_projection.weight)
                        label_data[label].append((lab_reps.cpu(), label_embeddings.cpu(), desc_input_ids.cpu(), desc_attention_mask.cpu()))

            # Now lets save the label_data
            pickle.dump(label_data, open(sys.argv[5],'wb'))
    
    if embedInstances:
        dt = raw_datasets['test'].with_transform(preprocess_function)
        testloader = DataLoader(raw_datasets['test'], collate_fn = default_data_collator, batch_size = training_args.per_device_eval_batch_size)

        # Lets get all the input documents
        inp_data = []#dict()
        import h5py
        f = h5py.File(sys.argv[5],'w')
        with torch.no_grad():
            ii = 0
            for item in tqdm(testloader):
                ii+=1
                batch_item = dict()
                batch_item['input_ids'] = item['input_ids'].cuda()
                batch_item['attention_mask'] = item['attention_mask'].cuda()
                batch_item['token_type_ids'] = item['token_type_ids'].cuda()
                outputs_doc, logits = model.forward_input_encoder(**batch_item)
                doc_reps = model.tok_proj(outputs_doc.last_hidden_state)
                for i in range(item['input_ids'].shape[0]):
                    igrp = f.create_group(f'{(ii-1) * item["input_ids"].shape[0] + i}')
                    igrp.create_dataset('logits', data = logits[i].detach().cpu().numpy())
                    igrp.create_dataset('doc_reps', data = doc_reps[i].detach().cpu().numpy())
                    igrp.create_dataset('labels', data = item['labels'][i].numpy())
                    igrp.create_dataset('input_ids', data = item['input_ids'][i].numpy())
        f.close()


print(sys.argv)
main()