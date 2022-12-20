from typing import Optional
from dataclasses import dataclass, field
from .constants import task_to_keys
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "The learning rate for the output encodeer of the model."}
    )
    place_model_on_device: Optional[bool] = field(
        default=True,
        metadata={"help" : "Useful if doing hyperparam search"}
    )
    scenario: Optional[str] = field(
        default="seen", # Options: seen, unseen_labels
        metadata={"help": "The scenario to use for training."}
    )

    one_hour_job : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Incase its a sequence of jobs, we will do advance management of checkpoints."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    all_labels : Optional[str] = field(
        default=None,
        metadata={"help": "The file containing all the labels. Mandatory if doing unseen labels"}
    )

    test_labels : Optional[str] = field(
        default=None,
        metadata={"help": "The file containing all the test labels."}
    )

    max_descs_per_label : Optional[int] = field(
        default = 999999,
        metadata={"help": "Restrict number of descriptions to be included per label"}
    )

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    load_from_local: bool = field(
        default=False,
        metadata={"help": "Whether to load the dataset from local or not."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    label_max_seq_length: int = field(default=32)
    contrastive_learning_samples : Optional[int] = field(
        default=-1,
        metadata={"help": "Number of samples to use for contrastive learning."},
    )
    cl_min_positive_descs : Optional[int] = field(
        default=20,
        metadata={"help": "Minimum number of positive descriptions to use for contrastive learning."},
    )
    descriptions_file : Optional[str] = field(
        # default='datasets/EUR-Lex/all_descriptions.json',
        default='datasets/EUR-Lex/eurlex-4k-class_descriptions_v1.json',
        metadata={"help": "A json file containing the descriptions."},
    )
    test_descriptions_file : Optional[str] = field(
        default='', # If empty, automatically make equal to descriptions_file
        metadata={"help": "A json file containing the test descriptions."},
    )


    cluster_path: Optional[str] = field(
        default='datasets/EUR-Lex/label_group_lightxml_0.npy',
        metadata={"help": "Path to the cluster file."},
    )
    num_clusters: int = field(
        default=64,
        metadata={"help": "Number of clusters in the cluster file."},
    )
    hyper_search: bool = field(
        default=False,
        metadata={"help": "Perform Hp Search"},
    )

    bm_short_file: str = field(
        default  = '',
        metadata = {"help": "BM Shortlist File to use for contrastive sampling"}
    )

    large_dset: bool = field(
        default = False,
        metadata = {"help" : "Dataset is modified in a way such that whole train set is not loaded"}
    )

    tokenized_descs_file: bool = field(
        default = False,
        metadata = {"help" : "Load the precomputed tokenized descriptions to speed up the process"}
    )

    train_tfidf_short: str = field(
        default = '',
        metadata = {"help" : "Shortlists based on the tf-idf values"}
    )

    test_tfidf_short: str = field(
        default = '',
        metadata = {"help" : "Shortlists based on the tf-idf values"}
    )

    ignore_pos_labels_file : str = field(
        default = '',
        metadata = {"help" : "Useful in fs setting"}
    )

    tok_format: int = field(
        default = -1,
        metadata = {"help" : "Tokenized Format for large datasets"}
    )

    coil_cluster_mapping_path : str = field(
        default = '',
        metadata = {"help" : "Clustering for coil matching based on BERT"}
    )

    random_sample_seed: int = field(
        default=-1,
        metadata={"help": "Random seed for eval sampling"},
    )
    
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    negative_sampling: Optional[str] = field(
        default="none",
        metadata={"help": "Whether to use negative sampling or not. Can be either `lightxml` or `none`."},
    )
    semsup : Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use semantic supervision or not."},
    )
    label_model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={"help": "The name or path of the label model to use."},
    )
    encoder_model_type: Optional[str] = field(
        default = 'bert',
        metadata={"help": "Type of encoder to use. Options: bert, roberta, xlnet"},
    )
    use_custom_optimizer: Optional[str] = field(
        default=None,
        metadata={"help": "Custom optimizer to use. Options: adamw"},
    )
    arch_type: Optional[int] = field(
        default=2,
        metadata={"help": '''Model architecture to use. Options: 1,2,3.\n1 -> LightXML Based\n2 -> No hidden layer\n3 -> Smaller Embedding Size'''},
    )
    devise: Optional[bool] = field(
        default = False,
        metadata = {"help" : 'Use Device Baseline'}
    )
    add_label_name : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Adds label name in beginning of all descriptions"}
    )

    normalize_embeddings : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Normalize Embeddings of input and output encoders before inner product."}
    )

    tie_weights : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Tie the Input & Label Transformer Weights(First 11 Layers) ."}
    )

    coil : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Use COILBert Variant"}
    )

    colbert: Optional[bool] = field(
        default  = False,
        metadata = {"help" : "Use COLBert, Note: coil must be set true"}
    )

    use_precomputed_embeddings : Optional[str] = field(
        default = '',
        metadata = {"help" : "PreComputed Embeddings Upto Level 9 of Bert for descriptions"}
    )

    token_dim : Optional[int] = field(
        default = 16,
        metadata = {"help": "Token Dimension for COILBert"}
    )

    pretrained_model_path : Optional[str] = field(
        default  = '',
        metadata = {"help" : "Use Pretrained Model for Finetuning (few shot setting)"}
    )
    pretrained_label_model_path : Optional[str] = field(
        default  = '',
        metadata = {"help" : "Use Pretrained Label Model for Finetuning (few shot setting)"}
    )
    

    num_frozen_layers : Optional[int] = field(
        default = 0,
        metadata = {
            "help" : "Freeze Input Encoder Layer"
        }
    )

    label_frozen_layers : Optional[int] = field(
        default = 0,
        metadata = {
            "help" : "Freeze Input Encoder Layer"
        }
    )