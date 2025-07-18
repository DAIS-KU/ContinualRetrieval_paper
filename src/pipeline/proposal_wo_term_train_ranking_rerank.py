import pickle
import random
import time
from typing import List

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from ablation import (
    Cluster,
    assign_instance_or_add_cluster,
    clear_invalid_clusters,
    evict_clusters,
    initialize,
    make_cos_query_psuedo_answers,
    retrieve_top_k_docs_from_cluster,
    get_samples_top_bottom,
    clear_unused_documents,
    Stream,
    DiversityBufferManager,
)
from data import read_jsonl, read_jsonl_as_dict, write_file, write_line
from functions import (
    InfoNCELoss,
    InfoNCETermLoss,
    evaluate_dataset,
    get_top_k_documents,
    renew_data_mean_pooling,
    get_top_k_documents_by_cosine,
)

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained(
    "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
)

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def encode_texts(model, texts, max_length=256):
    device = model.device
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    embedding = outputs[:, 0, :]  # [CLS]만 사용
    return embedding


def merge_nested_dicts(d1, d2):
    result = defaultdict(lambda: defaultdict(list))
    for d in [d1, d2]:
        for cluster_id, qid_dict in d.items():
            for qid, lengths in qid_dict.items():
                result[cluster_id][qid].extend(lengths)
    # defaultdict -> dict 변환
    return {k: dict(v) for k, v in result.items()}


def build_query_caches(clusters, docs, query_result):
    all_qids = []
    for cluster in clusters:
        all_qids.extend(cluster.get_only_qids(docs))
    q_caches = build_cluster_cache_table_by_cosine(
        qids=all_qids,
        clusters=clusters,
        docs=docs,
        # query_result=query_result,
        query_batch_size=64,
        doc_batch_size=512,
    )
    return all_qids, q_caches


def add_positive_caches(all_qids, q_caches, clusters, docs):
    all_pids = []
    for qid in all_qids:
        pids, _ = get_samples_top_bottom_3_with_cache(
            caches=q_caches,
            query=docs[qid],
            docs=docs,
            clusters=clusters,
            positive_k=1,
            negative_k=1,
            ts=None,
            use_tensor_key=True,
        )
        all_pids.extend(pids)
    print(
        f"build_positive_caches | all_qids: {len(all_qids)}, all_pids: {len(all_pids)}"
    )
    p_caches = build_cluster_cache_table_by_cosine(
        qids=all_pids,
        clusters=clusters,
        docs=docs,
        query_batch_size=64,
        doc_batch_size=512,
    )
    total_ids = all_qids + all_pids
    final_caches = merge_nested_dicts(q_caches, p_caches)
    return total_ids, final_caches


def streaming_train(
    queries,
    docs,
    ts,
    clusters: List[Cluster],
    model,
    num_epochs,
    positive_k=1,
    negative_k=6,
    learning_rate=2e-5,
    batch_size=32,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
):
    query_cnt = len(queries)
    loss_fn = InfoNCELoss()
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            query_batch = []
            pos_docs_all, neg_docs_all = [], []  # flat list of all pos/neg texts
            for idx in range(start_idx, end_idx):
                query = queries[idx]
                pos_ids, neg_ids = get_samples_top_bottom(
                    model=model,
                    query=query,
                    docs=docs,
                    clusters=clusters,
                    positive_k=positive_k,
                    negative_k=negative_k,
                )
                pos_docs = [docs[_id]["text"] for _id in pos_ids]
                neg_docs = [docs[_id]["text"] for _id in neg_ids]

                query_batch.append(query["query"])
                pos_docs_all.extend(pos_docs)  # flatten
                neg_docs_all.extend(neg_docs)  # flatten
            # batch_size 개 쿼리 → (batch_size, embedding_dim)
            query_embeddings = encode_texts(model=model, texts=query_batch)
            # 전체 긍정/부정 텍스트 → (batch_size * K, embedding_dim)
            pos_embeddings = encode_texts(model=model, texts=pos_docs_all)
            neg_embeddings = encode_texts(model=model, texts=neg_docs_all)
            # → (batch_size, K, embedding_dim)
            positive_embeddings = pos_embeddings.view(
                -1, positive_k, pos_embeddings.shape[-1]
            )
            negative_embeddings = neg_embeddings.view(
                -1, negative_k, neg_embeddings.shape[-1]
            )
            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_values.append(loss.item())  # loss.item()
            batch_cnt += 1
            print(
                f"Processed {end_idx}/{query_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / batch_cnt:.4f}"
            )
        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )
    return loss_values, ts


def train(
    start_session_number=0,
    end_sesison_number=12,
    load_cluster=True,
    sampling_rate=None,
    sampling_size_per_query=100,
    num_epochs=1,
    batch_size=32,
    warmingup_rate="none",
    positive_k=1,
    negative_k=6,
    cluster_min_size=50,
    nbits=12,
    max_iters=3,
    init_k=12,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
    warming_up_method="stream_seed",
    required_doc_size=20,
):
    total_loss_values = []
    loss_values_path = "../data/loss/total_loss_values_proposal_wo_term.txt"
    required_doc_size = (
        required_doc_size if required_doc_size is not None else positive_k + negative_k
    )

    prev_docs, clusters = None, []
    diversity_buffer_manager = DiversityBufferManager()

    for session_number in range(start_session_number, end_sesison_number):
        time_values_path = f"../data/loss/total_time_values_proposal_datasetL_wo_term_{session_number}.txt"
        ts = session_number
        model = BertModel.from_pretrained(
            "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
        ).to(devices[-1])
        model_path = None
        if session_number != 0:
            print("Load last session model.")
            model_path = (
                f"../data/model/proposal_wo_term_session_{session_number-1}.pth"
            )
            model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        model.train()
        new_model_path = f"../data/model/proposal_wo_term_session_{session_number}.pth"

        print(f"Training Session {session_number}/{load_cluster}")
        stream = Stream(
            session_number=session_number,
            model_path=model_path,
            query_path=f"/home/work/retrieval/data/datasetM_large_share/train_session{session_number}_queries.jsonl",
            doc_path=f"/home/work/retrieval/data/datasetM_large_share/train_session{session_number}_docs.jsonl",
            warmingup_rate=warmingup_rate,
            sampling_rate=sampling_rate,
            prev_docs=prev_docs,
            sampling_size_per_query=sampling_size_per_query,
            warming_up_method=warming_up_method,
        )
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")
        # Initial : 매번 로드 or 첫 세션만 로드
        if (load_cluster or session_number == start_session_number) and (
            session_number > 0
        ):
            print(f"Load last sesion clusters, docs.")
            with open(f"../data/clusters_wo_term_{session_number-1}.pkl", "rb") as f:
                clusters = pickle.load(f)
                print("Cluster loaded.")
            with open(f"../data/prev_docs_wo_term_{session_number-1}.pkl", "rb") as f:
                prev_docs = pickle.load(f)
                print("Prev_docs loaded.")
            stream.docs.update(prev_docs)
            batch_start = 0
        else:
            if session_number == 0:
                start_time = time.time()
                if warming_up_method == "stream_seed":
                    init_k = (
                        int(np.log2(len(stream.stream_docs[0])))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        model,
                        stream.stream_docs[0],
                        stream.docs,
                        init_k,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.stream_docs[0])
                    batch_start = 1
                else:
                    raise NotImplementedError(
                        f"Unsupported warming_up_method: {warming_up_method}"
                    )
                end_time = time.time()
                print(
                    f"Spend {end_time-start_time} seconds for clustering({len(clusters)}, {initial_size}) warming up."
                )
            else:
                batch_start = 0

        # Assign stream batch
        for i in range(batch_start, len(stream.stream_docs)):
            print(f"Assign {i}th stream starts.")
            start_time = time.time()
            assign_instance_or_add_cluster(
                model=model,
                clusters=clusters,
                cluster_min_size=cluster_min_size,
                stream_docs=stream.stream_docs[i],
                docs=stream.docs,
                ts=ts,
                use_tensor_key=use_tensor_key,
            )
            # if i % 50 == 0:
            #     for j, cluster in enumerate(clusters):
            #         print(f"{j}th size: {len(cluster.doc_ids)}")
            end_time = time.time()
            print(f"Assign {i}th stream ended({end_time - start_time}sec).")

        # Remain only trainable clusters
        clusters = clear_invalid_clusters(clusters, stream.docs, required_doc_size)
        # Train
        train_queries = stream.queries
        # print(
        #     f"=================================================BUILD QUERY CACHES======================================================"
        # )
        # start_time = time.time()
        # all_qids, q_caches = build_query_caches(clusters, stream.docs, query_result)
        # end_time = time.time()
        # print(
        #     f"==============================================DONE({end_time-start_time}sec)===================================================="
        # )
        # write_line(
        #     time_values_path, f"BuildQueryCaches({end_time-start_time}sec)\n", "a"
        # )
        # # Train
        # start_time = time.time()
        # train_queries = diversity_buffer_manager.get_samples(
        #     docs=stream.docs,
        #     clusters=clusters,
        #     caches=q_caches,
        #     sample_size=len(stream.queries),
        # )
        # end_time = time.time()

        # print(
        #     f"############################################QuerySelection({end_time - start_time}sec)############################################"
        # )
        # write_line(time_values_path, f"QuerySelection({end_time-start_time}sec)\n", "a")

        # print(
        #     f"=================================================BUILD POSITIVE CACHES======================================================"
        # )
        # start_time = time.time()
        # all_qids = [q["doc_id"] for q in train_queries]
        # total_ids, cluster_caches = add_positive_caches(
        #     all_qids, q_caches, clusters, stream.docs
        # )
        # end_time = time.time()
        # print(
        #     f"==============================================DONE({end_time-start_time}sec)===================================================="
        # )
        # write_line(
        #     time_values_path, f"BuildPositiveCaches({end_time-start_time}sec)\n", "a"
        # )

        loss_values, ts = streaming_train(
            queries=train_queries,
            docs=stream.docs,
            ts=ts,
            clusters=clusters,
            model=model,
            num_epochs=num_epochs,
            negative_k=negative_k,
            batch_size=batch_size,
            use_label=use_label,
            use_weight=use_weight,
        )
        write_line(
            loss_values_path, f"{session_number}, {', '.join(map(str, loss_values))}"
        )
        torch.save(model.state_dict(), new_model_path)
        # Evaluate
        # clusters, eval_stream_docs = evaluate_with_cluster(
        #     session_number=session_number,
        #     ts=ts,
        #     clusters=clusters,
        #     model_path=new_model_path,
        #     use_tensor_key=use_tensor_key,
        # )
        # evaluate(session_number=session_number, model_path=new_model_path)
        # Evict
        clusters = evict_clusters(model, stream.docs, clusters, ts, required_doc_size)
        stream.docs = clear_unused_documents(clusters, stream.docs)
        # Accumulate **eval_stream_docs
        prev_docs = stream.docs

        # with open(f"../data/clusters_wo_term_{session_number}.pkl", "wb") as f:
        #     pickle.dump(clusters, f)
        #     print("Cluster dumped.")
        # with open(f"../data/prev_docs_wo_term_{session_number}.pkl", "wb") as f:
        #     pickle.dump(prev_docs, f)
        #     print("Prev_docs dumped.")


def evaluate_with_cluster(
    session_number,
    ts,
    use_tensor_key,
    model_path,
    clusters: List[Cluster],
) -> List[Cluster]:
    eval_query_path = f"/home/work/retrieval/data/datasetM_large_share/test_session{session_number}_queries.jsonl"
    eval_doc_path = f"/home/work/retrieval/data/datasetM_large_share/test_session{session_number}_docs.jsonl"
    stream = Stream(
        session_number=session_number,
        query_path=eval_query_path,
        model_path=model_path,
        doc_path=eval_doc_path,
        warming_up_method="eval",
    )
    eval_query_count = len(stream.queries)
    eval_doc_count = len(stream.docs)
    print(
        f"Evaluate session {session_number} | #Query:{eval_query_count}, #Document:{eval_doc_count}"
    )
    # # Assign and Retrieve
    # start_time = time.time()
    # model = BertModel.from_pretrained("/home/work/retrieval/bert-base-uncased").to(
    #     devices[-1]
    # )
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.eval()
    # result = retrieve_top_k_docs_from_cluster(
    #     model, stream, clusters, use_tensor_key, 10
    # )
    # end_time = time.time()
    # print(f"Spend {end_time-start_time} seconds for retrieval.")

    # rankings_path = (
    #     f"../data/rankings/proposal_wo_term_{session_number}_with_cluster.txt"
    # )
    # write_file(rankings_path, result)
    # eval_log_path = f"../data/evals/proposal_wo_term_{session_number}_with_cluster.txt"
    # evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    return clusters, stream.docs


def model_builder(model_path):
    model = BertModel.from_pretrained("/home/work/retrieval/bert-base-uncased").to(
        devices[-1]
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate(session_number, model_path):
    method = "proposal_wo_term"
    print(f"Evaluate Session {session_number}")
    eval_query_path = f"/home/work/retrieval/data/datasetM_large_share/test_session{session_number}_queries.jsonl"
    eval_doc_path = f"/home/work/retrieval/data/datasetM_large_share/train_session{session_number}_docs.jsonl"

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    start_time = time.time()
    eval_query_data, eval_doc_data = renew_data_mean_pooling(
        model_builder, model_path, eval_query_data, eval_doc_data
    )
    result = get_top_k_documents_by_cosine(eval_query_data, eval_doc_data, 10)
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/{method}_{session_number}.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    del eval_query_data, eval_doc_data
