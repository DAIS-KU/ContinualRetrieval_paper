from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from .similarities import calculate_S_qd_regl_batch, calculate_S_qd_regl_batch_batch

device_count = torch.cuda.device_count()


def get_top_k_documents_gpu_cosine(
    new_q_data, docs, query_id, k, devices, batch_size=2048
):
    query_data_item = new_q_data[query_id]
    query_emb = query_data_item["EMB"]
    return get_top_k_documents_cosine(query_emb, docs, k, devices)


def get_top_k_documents_cosine(query_emb, docs, k, devices, batch_size=2048):
    docs_cnt = len(docs)
    num_gpus = len(devices)
    batch_indices = [
        list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
        for i in range((docs_cnt + batch_size - 1) // batch_size)
    ]
    gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

    def process_cosine(query_emb, gpu_batch_indices, device):
        query_emb = query_emb.to(device)
        scores = []
        for batch in gpu_batch_indices:
            start_idx, end_idx = batch[0], batch[-1] + 1
            print(f"{device}| Processing batch {start_idx}-{end_idx}")
            combined_embs = torch.stack(
                [doc["EMB"] for doc in docs[start_idx:end_idx]], dim=0
            ).to(device)
            score = F.cosine_similarity(
                # query_emb.unsqueeze(1), combined_embs.unsqueeze(0), dim=-1
                query_emb.unsqueeze(0),
                combined_embs,
                dim=-1,
            ).squeeze()
            # print(f"score: {score.shape}")
            scores.extend(
                [
                    (doc["doc_id"], score[idx].item())
                    for idx, doc in enumerate(docs[start_idx:end_idx])
                ]
            )
        return scores

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        args = [(query_emb, gpu_batch_indices[i], devices[i]) for i in range(num_gpus)]
        results = list(executor.map(lambda p: process_cosine(*p), args))

    combined_scores = [score for gpu_scores in results for score in gpu_scores]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    top_k_docs = combined_scores[:k]
    top_k_doc_ids = [x[0] for x in top_k_docs]
    return top_k_doc_ids


def get_top_k_documents_by_cosine(new_q_data, new_d_data, k=10, batch_size=2048):
    device_count = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
    print(f"Using GPUs: {devices}")

    docs = list(new_d_data.values())
    results = {}
    for qcnt, query_id in enumerate(new_q_data.keys()):
        results[query_id] = get_top_k_documents_gpu_cosine(
            new_q_data, docs, query_id, k, devices, batch_size
        )
        if qcnt % 10 == 0:
            print(f"#{qcnt} retrieving is done.")
    return results


# def get_top_k_documents_gpu(new_q_data, docs, query_id, k, devices, batch_size=2048):
#     query_data_item = new_q_data[query_id]
#     query_token_embs = query_data_item["TOKEN_EMBS"]

#     docs_cnt = len(docs)
#     num_gpus = len(devices)
#     batch_indices = [
#         list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
#         for i in range((docs_cnt + batch_size - 1) // batch_size)
#     ]
#     gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

#     def process(query_token_embs, gpu_batch_indices, device):
#         query_token_embs = query_token_embs.unsqueeze(0).to(device)
#         regl_scores = []
#         for batch in gpu_batch_indices:
#             start_idx, end_idx = batch[0], batch[-1] + 1
#             print(f"{device}| Processing batch {start_idx}-{end_idx}")
#             combined_embs = torch.stack(
#                 [doc["TOKEN_EMBS"].to(device) for doc in docs[start_idx:end_idx]], dim=0
#             )
#             # print(f'query_token_embs:{query_token_embs.shape}, combined_embs:{combined_embs.shape}')
#             regl_score = calculate_S_qd_regl_batch(
#                 query_token_embs, combined_embs, device
#             )
#             regl_scores.extend(
#                 [
#                     (doc["doc_id"], regl_score[idx].item())
#                     for idx, doc in enumerate(docs[start_idx:end_idx])
#                 ]
#             )
#         return regl_scores

#     with ThreadPoolExecutor(max_workers=num_gpus) as executor:
#         args = [
#             (query_token_embs, gpu_batch_indices[i], devices[i])
#             for i in range(num_gpus)
#         ]
#         results = list(executor.map(lambda p: process(*p), args))

#     combined_regl_scores = [score for gpu_scores in results for score in gpu_scores]
#     combined_regl_scores = sorted(
#         combined_regl_scores, key=lambda x: x[1], reverse=True
#     )
#     top_k_regl_docs = combined_regl_scores[:k]
#     top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]

#     return top_k_regl_doc_ids


# def get_top_k_documents(new_q_data, new_d_data, k=10, batch_size=2048):
#     device_count = torch.cuda.device_count()
#     devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
#     print(f"Using GPUs: {devices}")

#     docs = list(new_d_data.values())
#     results = {}
#     for qcnt, query_id in enumerate(new_q_data.keys()):
#         results[query_id] = get_top_k_documents_gpu(
#             new_q_data, docs, query_id, k, devices, batch_size
#         )
#         if qcnt % 10 == 0:
#             print(f"#{qcnt} retrieving is done.")
#     return results


def get_top_k_documents_gpu_batch(
    new_q_data_batch, docs, query_ids, k, devices, batch_size=2048
):
    queries_token_embs = [
        new_q_data_batch[qid]["TOKEN_EMBS"].to(devices[-1]) for qid in query_ids
    ]
    queries_token_embs = torch.stack(
        queries_token_embs, dim=0
    )  # (qbatch_size, seqlen, hidden)
    docs_cnt = len(docs)
    num_gpus = len(devices)
    batch_indices = [
        list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
        for i in range((docs_cnt + batch_size - 1) // batch_size)
    ]
    gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

    def process(queries_token_embs, gpu_batch_indices, device):
        queries_token_embs = queries_token_embs.to(device)
        all_regl_scores = [[] for _ in range(queries_token_embs.size(0))]  # 쿼리마다 별도로 저장
        for batch in gpu_batch_indices:
            start_idx, end_idx = batch[0], batch[-1] + 1
            print(f"{device}| Processing batch {start_idx}-{end_idx}")
            combined_embs = torch.stack(
                [doc["TOKEN_EMBS"].to(device) for doc in docs[start_idx:end_idx]], dim=0
            )  # (batch_size, seqlen, hidden)
            regl_score = calculate_S_qd_regl_batch_batch(
                queries_token_embs, combined_embs, device
            )  # (qbatch_size, batch_size)
            for q_idx in range(queries_token_embs.size(0)):
                all_regl_scores[q_idx].extend(
                    [
                        (
                            docs[start_idx + d_idx]["doc_id"],
                            regl_score[q_idx, d_idx].item(),
                        )
                        for d_idx in range(end_idx - start_idx)
                    ]
                )
        return all_regl_scores

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        args = [
            (queries_token_embs, gpu_batch_indices[i], devices[i])
            for i in range(num_gpus)
        ]
        results = list(executor.map(lambda p: process(*p), args))
    # 각 쿼리별로 결과 합치기
    qbatch_size = len(query_ids)
    combined_regl_scores_per_query = [[] for _ in range(qbatch_size)]
    for gpu_result in results:
        for q_idx, scores in enumerate(gpu_result):
            combined_regl_scores_per_query[q_idx].extend(scores)
    top_k_results = {}
    for q_idx, query_id in enumerate(query_ids):
        combined_regl_scores = combined_regl_scores_per_query[q_idx]
        combined_regl_scores = sorted(
            combined_regl_scores, key=lambda x: x[1], reverse=True
        )
        top_k_regl_docs = combined_regl_scores[:k]
        top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
        top_k_results[query_id] = top_k_regl_doc_ids
    return top_k_results


def get_top_k_documents(new_q_data, new_d_data, k=10, batch_size=2048, qbatch_size=10):
    device_count = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
    print(f"Using GPUs: {devices}")
    docs = list(new_d_data.values())
    query_ids = list(new_q_data.keys())
    results = {}
    for i in range(0, len(query_ids), qbatch_size):
        batch_query_ids = query_ids[i : i + qbatch_size]
        batch_q_data = {qid: new_q_data[qid] for qid in batch_query_ids}
        batch_results = get_top_k_documents_gpu_batch(
            batch_q_data, docs, batch_query_ids, k, devices, batch_size
        )
        results.update(batch_results)
        print(f"# {i} ~ {i+len(batch_query_ids)-1} retrieving is done.")
    return results
