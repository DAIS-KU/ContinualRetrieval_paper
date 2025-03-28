import os
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    target_model_path: str = field(
        default=None, metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"},
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
            "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    # 模型更新时是否考虑表达兼容
    compatible: bool = field(default=False)

    buffer_data: str = field(default=None)
    query_data: str = field(default=None)  # 如果是考虑表达兼容，就传embedding data，否则就传raw data
    doc_data: str = field(default=None)  # 如果是考虑表达兼容，就传embedding data，否则就传raw data
    mem_size: int = field(default=30)
    mem_batch_size: int = field(default=2)
    cl_method: str = field(default=None)
    retrieve_method: str = field(default="random")
    update_method: str = field(default="random")

    # ### MIR
    subsample: int = field(default=0)

    # ### GSS
    gss_mem_strength: int = field(default=0)
    gss_batch_size: int = field(default=0)

    # ### our method
    new_batch_size: int = field(default=2)  # 用于memory retrieval
    alpha: float = field(default=0.6)  # 用于memory retrieval
    beta: float = field(default=0.4)  # 用于memory retrieval
    gamma: float = field(default=0.0)  # 用于memory retrieval ocs 수정
    mem_upsample: int = field(default=6)  # 用于memory retrieval
    mem_eval_size: int = field(default=10)  # 用于memory update
    mem_replace_size: int = field(default=10)  # 用于memory update
    upsample_scale: float = field(default=2.0)  # 用于memory update

    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=" ")
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"}
    )
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )

    encode_in_path: List[str] = field(
        default=None, metadata={"help": "Path to data to encode"}
    )
    encoded_save_path: str = field(
        default=None, metadata={"help": "where to save the encode"}
    )
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    seq_max_len: int = field(
        default=160,
        metadata={
            "help": "The maximum total input sequence length after tokenization for sequence. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the data downloaded from huggingface"
        },
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split("/")
            self.dataset_split = info[-1] if len(info) == 3 else "train"
            self.dataset_name = (
                "/".join(info[:-1]) if len(info) == 3 else "/".join(info)
            )
            self.dataset_language = "default"
            if ":" in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(":")
        else:
            self.dataset_name = "json"
            self.dataset_split = "train"
            self.dataset_language = "default"
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith("jsonl") or f.endswith("json")
            ]
        else:
            self.train_path = None


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(
        default=False, metadata={"help": "share negatives across devices"}
    )
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"}
    )
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    compatible_ce_alpha: float = field(default=0.0)
