#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
t('The script has began')
import itertools
import logging
import os
import random
import sys
import time
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
from src import SemSupDataset
from src import AutoModelForMultiLabelClassification
from src import multilabel_metrics
from src import task_to_keys, task_to_label_keys, dataset_to_numlabels
from src import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from src import dataset_classification_type
from src import BertForSemanticEmbedding
from src import read_yaml_config
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)




def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def get_last_check(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

def main():


    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    print('Main Function is Called!!!', sys.argv)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    
    if len(sys.argv) > 1 and sys.argv[1].startswith('--local_rank'):
        extra_args = {'local_rank' : sys.argv[1].split('=')[1]}
        argv = sys.argv[0:1] + sys.argv[2:]
    else:
        argv = sys.argv
        extra_args = {}

    print(len(argv) == 3 and argv[1].endswith(".yml"))
    if len(argv) == 2 and argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(argv[1]))
    elif len(argv) == 3 and argv[1].endswith(".yml"):
        model_args, data_args, training_args = parser.parse_dict(read_yaml_config(os.path.abspath(argv[1]), output_dir = argv[2],  extra_args = extra_args))
        print('training args', training_args)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)
    if training_args.seed == -1:
        training_args.seed = np.random.randint(0, 100000000)
    print(training_args.seed)


    last_checkpoint = get_last_check(training_args)

    set_seed(training_args.seed)

    
    if data_args.dataset_name is not None and not data_args.load_from_local:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            print('df are', data_files, model_args.cache_dir)
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Labels
    if data_args.task_name is not None:
        label_key = task_to_label_keys[data_args.task_name] 


        if training_args.scenario == 'unseen_labels':
            label_list = [x.strip() for x in open(data_args.all_labels).readlines()]
            train_labels = list(set([item for sublist in raw_datasets['train'][label_key] for item in sublist]))
            if data_args.test_labels is not None:
                test_labels = [x.strip() for x in open(data_args.test_labels).readlines()]
            else:
                test_labels  = list(set([item for sublist in raw_datasets['validation'][label_key] for item in sublist]))

        else:
            label_list = list(set(itertools.chain(*[
                [item for sublist in raw_datasets[split_key][label_key] for item in sublist]
                for split_key in raw_datasets.keys()]
            )))
        num_labels = len(label_list)
        label_list.sort() # For consistency
        print('Debugging: num_labels: ', num_labels)
        print('Debugging: label_list[:50]: ', label_list[:50])
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.semsup:
        label_model, label_tokenizer = getLabelModel(data_args, model_args)
        
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.model_name_or_path = model_args.model_name_or_path
    config.problem_type = dataset_classification_type[data_args.task_name]
    config.negative_sampling = model_args.negative_sampling
    config.semsup = model_args.semsup
    config.encoder_model_type = model_args.encoder_model_type
    config.arch_type = model_args.arch_type
    config.coil = model_args.coil
    config.token_dim = model_args.token_dim
    config.colbert = model_args.colbert

    if config.semsup:
        config.label_hidden_size = label_model.config.hidden_size
        print('Label hidden size is ', label_model.config.hidden_size)

    temp_label_id = {v: i for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,#model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

        
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    def model_init():
        model = BertForSemanticEmbedding(config)
        num_frozen_layers = model_args.num_frozen_layers
        if num_frozen_layers > 0:
            try:
                for param in model.encoder.bert.embeddings.parameters():
                    param.requires_grad = False
                for param in model.encoder.bert.pooler.parameters():
                    param.requires_grad = False
                for layer in model.encoder.bert.encoder.layer[:num_frozen_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
            except:
                for param in model.encoder.embeddings.parameters():
                    param.requires_grad = False
                for param in model.encoder.pooler.parameters():
                    param.requires_grad = False
                for layer in model.encoder.encoder.layer[:num_frozen_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Place the label model inside the main model
        if model_args.semsup:
            model.label_model = label_model
            model.label_tokenizer = label_tokenizer
            if model_args.tie_weights:
                for i in range(9):
                    if num_frozen_layers >= 9: 
                        try:
                            model.label_model.encoder.layer[i]  = model.encoder.bert.encoder.layer[i]
                        except:
                            model.label_model.encoder.layer[i]  = model.encoder.encoder.layer[i]
                    else:
                        for param in model.label_model.encoder.layer[i].parameters():
                            param.requires_grad = False
                for param in model.label_model.embeddings.parameters():
                    param.requires_grad = False
                for param in model.label_model.pooler.parameters():
                    param.requires_grad = False
            else:
                label_frozen_layers = model_args.label_frozen_layers
                if label_frozen_layers > 0:
                    print(model.label_model)
                    for param in model.label_model.embeddings.parameters():
                        param.requires_grad = False
                    for param in model.label_model.pooler.parameters():
                        param.requires_grad = False
                    for layer in model.label_model.encoder.layer[:label_frozen_layers]:
                        for param in layer.parameters():
                            param.requires_grad = False

        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
        return model
    model = model_init()
    if model_args.pretrained_model_path != '':
        model.load_state_dict(torch.load(model_args.pretrained_model_path, map_location = list(model.parameters())[0].device))
    if model_args.pretrained_label_model_path != '':
        model.label_model.load_state_dict(torch.load(model_args.pretrained_label_model_path, map_location = list(model.parameters())[0].device))

    id2label = model.config.id2label
    label_to_id = model.config.label2id

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and label_key in examples:
            # check if multi-label problem
            if isinstance(examples[label_key][0], list):
                # Multi-Label, create one-hot encoding
                labels = [[label_to_id[l] for l in examples[label_key][i]] for i in range(len(examples[label_key]))]
                result["label"] = [[1 if j in labels[i] else 0 for j in range(num_labels)] for i in range(len(labels))]
            else:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        

        # Labels keyword should not be present(it may contain string)
        try: del input['labels']
        except: ...

        return result

    try:
        if data_args.test_descriptions_file == '':
            data_args.test_descriptions_file = data_args.descriptions_file
    except: data_args.test_descriptions_file = data_args.descriptions_file
    
    print('Running with_transform')
    raw_datasets = raw_datasets.with_transform(preprocess_function)

    class_descs_tokenized  = None
    if model_args.semsup and data_args.large_dset and os.path.exists(data_args.tokenized_descs_file):
        if data_args.tokenized_descs_file.endswith('npy'):
            class_descs_tokenized = np.load(data_args.tokenized_descs_file, allow_pickle=True)

        
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(np.random.choice(len(train_dataset), max_train_samples))
        if model_args.semsup:
            train_dataset = SemSupDataset(train_dataset, data_args, data_args.descriptions_file, label_to_id, id2label, label_tokenizer, return_desc_embeddings = True, sampleRandom = data_args.contrastive_learning_samples, cl_min_positive_descs= data_args.cl_min_positive_descs, seen_labels = None if training_args.scenario == 'seen' else train_labels, add_label_name = model_args.add_label_name, max_descs_per_label = data_args.max_descs_per_label, use_precomputed_embeddings = model_args.use_precomputed_embeddings, bm_short_file = data_args.bm_short_file, ignore_pos_labels_file = data_args.ignore_pos_labels_file, class_descs_tokenized = class_descs_tokenized)
        else:
            train_dataset = SemSupDataset(train_dataset, data_args, data_args.descriptions_file, label_to_id, id2label, None, useSemSup = False, add_label_name = model_args.add_label_name)

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        choice_indexes = None

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            if data_args.random_sample_seed != -1:
                l = len(eval_dataset)

                np.random.seed(data_args.random_sample_seed)
                choice_indexes = np.random.choice(l, max_eval_samples, replace = False).tolist()
                choice_indexes = [x for x in choice_indexes]
                import pickle
                pickle.dump(choice_indexes, open('choice_indexes.pkl','wb'))
                eval_dataset = eval_dataset.select(choice_indexes)
                np.random.seed()
            else:
                choice_indexes = None
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        if model_args.semsup:
            eval_dataset = SemSupDataset(eval_dataset, data_args, data_args.test_descriptions_file, label_to_id, id2label, label_tokenizer, return_desc_embeddings=True, seen_labels = None if training_args.scenario == 'seen' else test_labels, add_label_name = model_args.add_label_name, max_descs_per_label = data_args.max_descs_per_label, use_precomputed_embeddings = model_args.use_precomputed_embeddings, class_descs_tokenized = class_descs_tokenized, isTrain = False, choice_indexes = choice_indexes)

    if training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))


    compute_metrics = multilabel_metrics(data_args, model.config.id2label, model.config.label2id, {}, training_args)

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    

    # Initialize our Trainer
    print('Initializing Optimizers')
    from torch.optim import AdamW
    # from transformers import AdamW
    if model_args.use_custom_optimizer:
        decay_cond = lambda x: x[0].lower().find('layernorm.weight')!=-1 or x[0].lower().find('bias')!=-1
        if model_args.semsup and not data_args.hyper_search:
            main_decay_params = list(map(lambda x: x[1], filter(lambda x: x[1].requires_grad and decay_cond(x) and (x[0][12:], x[1]) not in model.label_model.named_parameters() , model.named_parameters())))
            main_no_decay_params = list(map(lambda x: x[1],filter(lambda x: x[1].requires_grad and not decay_cond(x) and (x[0][12:], x[1]) not in model.label_model.named_parameters(), model.named_parameters())))
            label_decay_params = list(map(lambda x: x[1], filter(lambda x: x[1].requires_grad and decay_cond(x) , model.label_model.named_parameters())))
            label_no_decay_params = list(map(lambda x: x[1],filter(lambda x: x[1].requires_grad and not decay_cond(x), model.label_model.named_parameters())))
            if model_args.tie_weights:
                label_decay_params = list(set(label_decay_params).difference(main_decay_params))
                label_no_decay_params = list(set(label_no_decay_params).difference(main_no_decay_params))
            
            optimizer = AdamW([
                {'params': main_decay_params, 'weight_decay': 1e-2},
                {'params': main_no_decay_params, 'weight_decay': 0},
                {'params': label_decay_params, 'weight_decay': 1e-2, 'lr' : training_args.output_learning_rate},
                {'params': label_no_decay_params, 'weight_decay': 0, 'lr' : training_args.output_learning_rate}
            ],
            lr = training_args.learning_rate, eps= 1e-6)    
            ...
        else:
            decay_params = list(map(lambda x: x[1], filter(lambda x: decay_cond(x), model.named_parameters())))
            no_decay_params = list(map(lambda x: x[1],filter(lambda x: not decay_cond(x), model.named_parameters())))
            optimizer = optim.AdamW([
                {'params': decay_params, 'weight_decay': 1e-2},
                {'params': no_decay_params, 'weight_decay': 0}],
            lr = training_args.learning_rate, eps= 1e-6)
        

    trainer = Trainer(
        model=model,
        model_init= None,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers = (optimizer if model_args.use_custom_optimizer else None, None) ,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)
            trainer.log_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()