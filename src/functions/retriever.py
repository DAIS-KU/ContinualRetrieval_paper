import torch
from concurrent.futures import ThreadPoolExecutor


def get_passage_embeddings(model, passages, device, max_length=256):
    batch_inputs = tokenizer(
        passages,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    with torch.no_grad():
        outputs = model(**batch_inputs).last_hidden_state
    token_embeddings = outputs[:, 1:-1, :]
    attention_mask = batch_inputs["attention_mask"][:, 1:-1]
    token_embeddings = token_embeddings * (attention_mask[:, :, None].to(device))
    return token_embeddings


def get_top_k_documents(
    query, closest_cluster_id, cluster_instances, k, device, term_score=False
):
    candidate_k, top_k = k, k
    if term_score:
        candidate_k = 10 * k

    closest_cluster_instances = cluster_instances[closest_cluster_id]
    bm25, doc_ids = get_bm25(closest_cluster_instances)
    scores = bm25.get_scores(preprocess(query["TEXT"]))
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :candidate_k
    ]
    top_k_doc_ids = [doc_ids[i] for i in top_k_indices]
    if term_score:
        regl_scores = []
        combined_embs = torch.stack(
            [
                closest_cluster_instances[i]["TOKEN_EMBS"].to(device)
                for i in top_k_indices
            ],
            dim=0,
        )
        regl_score = calculate_S_qd_regl_batch(
            query["TOKEN_EMBS"].to(device), combined_embs, device
        )
        regl_scores = [
            (doc["ID"], regl_score[idx].item())
            for idx, doc in enumerate(closest_cluster_instances)
        ]
        top_k_regl_docs = sorted(regl_scores, key=lambda x: x[1], reverse=True)[:top_k]
        top_k_doc_ids = [x[0] for x in top_k_regl_docs]

    return top_k_doc_ids


def process_queries_on_gpu(
    gpu_id, query_data, query_batch_keys, centroids, cluster_instances, result, device
):
    device = torch.device(f"cuda:{gpu_id}")
    for qid in query_batch_keys:
        query = query_data[qid]
        closest_cluster_id = find_closest_cluster_id(query, centroids, device)
        top_k_doc_ids = get_top_k_documents(
            query, closest_cluster_id, cluster_instances, 10, device, False
        )
        result[qid] = top_k_doc_ids


def split_query_keys(query_data, num_gpus):
    keys = list(query_data.keys())
    chunk_size = (len(keys) + num_gpus - 1) // num_gpus
    return [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]


def process_queries_with_gpus(query_data, centroids, cluster_instances, devices):
    query_batch_keys = split_query_keys(query_data, len(devices))

    def process_on_gpu(args):
        gpu_id, query_keys = args
        return process_queries_on_gpu(
            gpu_id, query_data, query_keys, centroids, cluster_instances, devices
        )

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        results = list(
            executor.map(
                process_on_gpu,
                [
                    (gpu_id, query_keys)
                    for gpu_id, query_keys in enumerate(query_batch_keys)
                ],
            )
        )
    merged_result = defaultdict(list)
    for partial_result in results:
        for key, value in partial_result.items():
            merged_result[key].extend(value)

    print("Processing complete!")
    return merged_result
