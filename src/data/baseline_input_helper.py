import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import BatchEncoding, BertTokenizer

from .loader import read_jsonl, read_jsonl_as_dict, load_train_docs

max_q_len: int = 32
max_p_len: int = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py


def _prepare_inputs(
    session_number,
    inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...],
    buffer,
    cl_method,
    new_batch_size,
    mem_batch_size,
    compatible,
) -> List[Dict[str, Union[torch.Tensor, Any]]]:
    # tuple로 들어와야하는데 리스트고, 리스트안에 튜플 들어있음 어케 쎃아서 줘야햄,,,,
    # Trainer에서는 하나씩 퍼다나르나
    # print(f"inputs: {inputs}")
    prepared = []
    for x in inputs[2:]:
        for key, val in x.items():
            x[key] = val.to(device)
        prepared.append(x)

    if cl_method == "er":
        if not compatible or session_number == 0:
            qid_lst, docids_lst = inputs[0], inputs[1]
            # print(f"Before sampling: {docids_lst}")

            mem_docids_lst, mem_passage = buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
                # lr=self._get_learning_rate(),
            )  # ER: [num_q * mem_bz, d_len], cpu; MIR: [num_q, mem_bz, d_len], gpu
            buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    passage_len = val.size(-1)
                    prepared[1][key] = prepared[1][key].reshape(
                        len(qid_lst), -1, passage_len
                    )  # [num_q, bz, d_len]
                    val = val.reshape(len(qid_lst), -1, passage_len).to(
                        prepared[1][key].device
                    )  # [num_q, mem_bz, d_len]
                    prepared[1][key] = torch.cat(
                        (prepared[1][key], val), dim=1
                    ).reshape(
                        -1, passage_len
                    )  # [num_q*(bz+mem_bz), d_len]
            all_docids_lst = docids_lst + mem_docids_lst
            # print(f"After sampling: {all_docids_lst}")
            prepared.append(all_docids_lst)  # for updating old emb
        else:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_docids_lst, mem_passage = buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
                # lr=self._get_learning_rate(),
            )  # ER:[num_q * mem_bz],cpu, [num_q * mem_bz, d_len], cpu; MIR: [num_q, mem_bz],cpu, [num_q, mem_bz, d_len], gpu
            buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )

            if mem_passage is not None:
                # print(f"mem_docids_lst: {mem_docids_lst}, mem_passage:{mem_passage}")
                for key, val in mem_passage.items():
                    passage_len = val.size(-1)
                    prepared[1][key] = prepared[1][key].reshape(
                        len(qid_lst), -1, passage_len
                    )  # [num_q, bz, d_len]
                    val = val.reshape(len(qid_lst), -1, passage_len).to(
                        prepared[1][key].device
                    )  # [num_q, mem_bz, d_len]
                    prepared[1][key] = torch.cat(
                        (prepared[1][key], val), dim=1
                    ).reshape(
                        -1, passage_len
                    )  # [num_q*(bz+mem_bz), d_len]
                # docids_lst = torch.tensor(docids_lst).reshape(
                docids_lst = torch.stack(
                    [torch.tensor(docid) for docid in docids_lst]
                ).reshape(
                    len(qid_lst), -1
                )  # [num_q, n]
                mem_docids_lst = mem_docids_lst.reshape(
                    len(qid_lst), -1
                )  # [num_q, mem_bz]
                all_docids_lst = torch.cat(
                    (docids_lst, mem_docids_lst), dim=-1
                ).reshape(
                    -1
                )  # [num_q * n+mem_bz]

                identity = []
                doc_oldemb = []
                for i, docids in enumerate(all_docids_lst):
                    docids = int(docids)
                    if docids in buffer.buffer_did2emb:
                        identity.append(i)
                        doc_oldemb.append(buffer.buffer_did2emb[docids])
                print(
                    f"all_docids_lst: {all_docids_lst}, identity:{len(identity)}, buffer.buffer_did2emb:{len(buffer.buffer_did2emb.keys())}"
                )
                # identity =torch.tensor(identity)
                doc_oldemb = torch.stack(doc_oldemb).to(device)
                # doc_oldemb = torch.tensor(np.array(doc_oldemb), device=device)
                prepared.append(identity)
                prepared.append(doc_oldemb)
            prepared.append(all_docids_lst)  # for updating old emb
    elif cl_method == "mir":
        if not compatible or session_number == 0:
            qid_lst, docids_lst = inputs[0], inputs[1]

            docids_lst_from_mem, mem_passage = buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
                lr=1e-1,  # 5e-1, 1e-1, 1e-2, https://arxiv.org/pdf/1908.04742
            )  # ER: [num_q * mem_bz, d_len], cpu; MIR: [num_q, mem_bz, d_len], gpu
            buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    passage_len = val.size(-1)
                    prepared[1][key] = prepared[1][key].reshape(
                        len(qid_lst), -1, passage_len
                    )  # [num_q, bz, d_len]
                    val = val.reshape(len(qid_lst), -1, passage_len).to(
                        prepared[1][key].device
                    )  # [num_q, mem_bz, d_len]
                    prepared[1][key] = torch.cat(
                        (prepared[1][key], val), dim=1
                    ).reshape(
                        -1, passage_len
                    )  # [num_q*(bz+mem_bz), d_len]
            all_docids_lst = docids_lst + docids_lst_from_mem
            prepared.append(all_docids_lst)  # for updating old emb
            # print(f"prepared: {len(prepared)}, {len(prepared[0])}")
        else:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_docids_lst, mem_passage = buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
                lr=1e-1,  # 5e-1, 1e-1, 1e-2, https://arxiv.org/pdf/1908.04742
            )  # ER:[num_q * mem_bz],cpu, [num_q * mem_bz, d_len], cpu; MIR: [num_q, mem_bz],cpu, [num_q, mem_bz, d_len], gpu
            buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    passage_len = val.size(-1)
                    prepared[1][key] = prepared[1][key].reshape(
                        len(qid_lst), -1, passage_len
                    )  # [num_q, bz, d_len]
                    val = val.reshape(len(qid_lst), -1, passage_len).to(
                        prepared[1][key].device
                    )  # [num_q, mem_bz, d_len]
                    prepared[1][key] = torch.cat(
                        (prepared[1][key], val), dim=1
                    ).reshape(
                        -1, passage_len
                    )  # [num_q*(bz+mem_bz), d_len]

                docids_lst = torch.tensor(docids_lst).reshape(
                    len(qid_lst), -1
                )  # [num_q, n]
                mem_docids_lst = mem_docids_lst.reshape(
                    len(qid_lst), -1
                )  # [num_q, mem_bz]
                all_docids_lst = torch.cat(
                    (docids_lst, mem_docids_lst), dim=-1
                ).reshape(
                    -1
                )  # [num_q * n+mem_bz]

                identity = []
                doc_oldemb = []
                for i, docids in enumerate(all_docids_lst):
                    docids = int(docids)
                    if docids in buffer.buffer_did2emb:
                        identity.append(i)
                        doc_oldemb.append(buffer.buffer_did2emb[docids])
                # identity = torch.tensor(identity)
                doc_oldemb = torch.stack(doc_oldemb).to(device)
                # doc_oldemb = torch.tensor(np.array(doc_oldemb), device=device)
                prepared.append(identity)
                prepared.append(doc_oldemb)
            prepared.append(all_docids_lst)  # for updating old emb
            # print(f"prepared: {len(prepared)}, {len(prepared[0])}")
    elif cl_method == "our" or cl_method == "l2r":
        if not compatible or session_number == 0:
            qid_lst, docids_lst = inputs[0], inputs[1]
            # print(f"model {next(buffer.model.parameters()).device}, prepared[0] {prepared[0]['input_ids'].device}, prepared[1] {prepared[1]['input_ids'].device}")

            mem_passage, pos_docids, candidate_neg_docids = buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )  # [num_q*(new_bz+mem_bz), d_len]
            buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                pos_docids=pos_docids,
                candidate_neg_docids=candidate_neg_docids,
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    prepared[1][key] = val
            prepared.append(docids_lst)  # for updating old emb
        else:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_emb, mem_passage, pos_docids, candidate_neg_docids = buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )  # [num_q*(1+mem_bz), 768],gpu; [num_q*(new_bz+mem_bz), d_len],gpu
            buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                pos_docids=pos_docids,
                candidate_neg_docids=candidate_neg_docids,
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    prepared[1][key] = val

                identity = []  # [1+mem_batch_size, num_q]
                pos_identity = torch.arange(len(qid_lst)) * (
                    1 + new_batch_size + mem_batch_size
                )
                identity.append(pos_identity)
                for i in range(mem_batch_size):
                    identity.append(pos_identity + i + 1 + new_batch_size)
                identity = torch.stack(identity, dim=0).transpose(0, 1).reshape(-1)
                prepared.append(identity)
                prepared.append(mem_emb)
            prepared.append(docids_lst)  # for updating old emb
    elif cl_method == "incre":
        if compatible:
            qid_lst, docids_lst = inputs[0], inputs[1]
            docids_lst = torch.tensor(docids_lst).reshape(
                len(qid_lst), -1
            )  # [num_q, n]

            identity = torch.arange(docids_lst.size(0)) * docids_lst.size(1)
            prepared.append(identity)

            doc_oldemb = []  # [num_q, 768]
            for docid in docids_lst[:, 0]:  # 对于incre，只有正例是old doc
                doc_oldemb.append(buffer.buffer_did2emb[int(docid)])
            doc_oldemb = torch.tensor(np.array(doc_oldemb)).to(device)
            prepared.append(doc_oldemb)
            prepared.append(docids_lst)  # for updating old emb
    else:
        print("not implement...")

    return prepared


def create_one_example(text_encoding: List[int], is_query=False):
    item = tokenizer.encode_plus(
        text_encoding,
        truncation="only_first",
        max_length=max_q_len if is_query else max_p_len,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    # print(f"item: {item}")# input_ids만 포함된 딕셔너리.
    return item


def collate(features):
    # features 단일아이템 처리??
    qq_id, dd_id, qq, dd = [features[0]], [features[1]], [features[2]], [features[3]]
    # qq_id = [f[0] for f in features]
    # dd_id = [f[1] for f in features]
    # qq = [f[2] for f in features]
    # dd = [f[3] for f in features]

    if isinstance(qq_id[0], list):
        qq_id = sum(qq_id, [])
    if isinstance(dd_id[0], list):
        dd_id = sum(dd_id, [])
    if isinstance(qq[0], list):
        qq = sum(qq, [])
    if isinstance(dd[0], list):
        dd = sum(dd, [])

    # print(f"qq_id: {qq_id}")
    # print(f"dd_id: {dd_id}")
    # print(f"qq: {qq}")
    # print(f"dd: {dd}")
    q_collated = tokenizer.pad(
        qq,
        padding="max_length",
        max_length=max_q_len,
        return_tensors="pt",
    )
    d_collated = tokenizer.pad(
        dd,
        padding="max_length",
        max_length=max_p_len,
        return_tensors="pt",
    )
    # print(f"q_collated: {q_collated}")
    # print(f"d_collated: {d_collated}")
    return qq_id, dd_id, q_collated, d_collated


def build_bm25(docs):
    tokenized_corpus = [word_tokenize(doc["text"].lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def get_candidates(session_number, bm25, query, doc_ids):
    k = 500 if session_number == 0 else 200
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[-k:]
    top_k_doc_ids = [doc_ids[i] for i in top_k_indices]
    return top_k_doc_ids


def getitem(
    session_number, query, docs, filtered=False, bm25=None, train_n_passages=12
) -> Tuple[BatchEncoding, List[BatchEncoding]]:
    qry_id = query["qid"]
    qry = query["query"]
    encoded_query = create_one_example(qry, is_query=True)

    psg_ids = []
    encoded_passages = []

    pos_id = query["cos_ans_pids"][0]
    pos_psg = docs[pos_id]["text"]
    psg_ids.append(pos_id)
    encoded_passages.append(create_one_example(pos_psg))

    negative_size = train_n_passages - 1
    doc_ids = list(docs.keys())
    if filtered:
        neg_ids = get_candidates(session_number, bm25, qry, doc_ids)
    else:
        valid_neg_ids = list(set(doc_ids) - set(query["cos_ans_pids"]))
        neg_ids = random.sample(valid_neg_ids, negative_size)

    for neg_id in neg_ids:
        psg_ids.append(neg_id)
        encoded_passages.append(create_one_example(docs[neg_id]["text"]))
    return qry_id, psg_ids, encoded_query, encoded_passages


def load_inputs(
    session_number,
    query_path,
    doc_path,
    compatible,
    filtered=False,
):
    queries = read_jsonl(query_path, True, compatible)
    random.shuffle(queries)
    docs = load_train_docs(session_number)
    bm25 = build_bm25(list(docs.values())) if filtered else None
    inputs = []
    for query in queries:
        features = getitem(session_number, query, docs, filtered, bm25)
        _input = collate(features)
        inputs.append(_input)
    # print(f"load_inputs {type(inputs)}, {type(inputs[0])}, {len(inputs)}, {len(inputs[0])}")
    return inputs


# https://github.com/caiyinqiong/L-2R/blob/main/run_example.sh
def prepare_inputs(
    session_number,
    query_path,
    doc_path,
    buffer,
    cl_method,
    new_batch_size=2,
    mem_batch_size=2,
    compatible=False,
):
    print(f"Visit cl_method {cl_method}, compatible {compatible}")
    filtered = cl_method == "l2r"
    inputs = load_inputs(session_number, query_path, doc_path, compatible, filtered)
    prepared_inputs = []
    for _input in inputs:
        result = _prepare_inputs(
            session_number,
            _input,
            buffer,
            cl_method,
            new_batch_size,
            mem_batch_size,
            compatible,
        )
        prepared_inputs.append(result)
    # print(f"prepared_inputs {type(prepared_inputs)}, {type(prepared_inputs[0])}, {len(prepared_inputs)}, {len(prepared_inputs[0])}")
    return prepared_inputs
