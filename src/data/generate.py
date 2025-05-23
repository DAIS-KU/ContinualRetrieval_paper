import json
import random
from collections import Counter, defaultdict

from datasets import load_dataset

from .loader import append_to_jsonl, read_jsonl, read_jsonl_as_dict


def lotte_check_query_answer_docs(domains):
    for domain in domains:
        quries_path = f"./raw/merged/{domain}_queries.jsonl"
        queries = read_jsonl(quries_path)
        answer_doc_ids = set()
        total_answer_doc_cnt = 0
        for q in queries:
            answer_doc_ids.update(q["answer_pids"])
            total_answer_doc_cnt += len(q["answer_pids"])
        answer_cnt = len(answer_doc_ids)

        doc_path = f"./raw/merged/{domain}_docs.jsonl"
        doc_cnt = count_jsonl_elements(doc_path)

        print(
            f"{domain} : answer {answer_cnt} / total {doc_cnt} = {answer_cnt/doc_cnt * 100}%, total_answer_doc_cnt: {total_answer_doc_cnt}"
        )


# def sessioning(
#     dataset,
#     domains,
#     domain_answer_rate,
#     need_train_query_counts,
#     need_test_query_counts,
# ):
#     for domain, ans_rate in zip(domains, domain_answer_rate):
#         queries = read_jsonl(f"/home/work/retrieval/data/raw/{dataset}/{domain}_queries.jsonl", True)
#         random.shuffle(queries)
#         answer_train_doc_ids, answer_test_doc_ids = defaultdict(set), defaultdict(set)

#         print(
#             f"need_train_query_counts: {need_train_query_counts}, need_test_query_counts: {need_test_query_counts}"
#         )
#         train_cnt0, train_cnt1, train_cnt2, train_cnt3 = 0, 0, 0, 0
#         test_cnt0, test_cnt1, test_cnt2, test_cnt3 = 0, 0, 0, 0
#         for query in queries:
#             # TRAIN
#             if train_cnt0 < need_train_query_counts[0]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session0_queries.jsonl"
#                 train_cnt0 += 1
#                 answer_train_doc_ids[0].update(query["answer_pids"])
#             elif train_cnt1 < need_train_query_counts[1]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session1_queries.jsonl"
#                 train_cnt1 += 1
#                 answer_train_doc_ids[1].update(query["answer_pids"])
#             elif train_cnt2 < need_train_query_counts[2]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session2_queries.jsonl"
#                 train_cnt2 += 1
#                 answer_train_doc_ids[2].update(query["answer_pids"])
#             elif train_cnt3 < need_train_query_counts[3]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session3_queries.jsonl"
#                 train_cnt3 += 1
#                 answer_train_doc_ids[3].update(query["answer_pids"])
#             # TEST
#             elif test_cnt0 < need_test_query_counts[0]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session0_queries.jsonl"
#                 test_cnt0 += 1
#                 answer_test_doc_ids[0].update(query["answer_pids"])
#             elif test_cnt1 < need_test_query_counts[1]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session1_queries.jsonl"
#                 test_cnt1 += 1
#                 answer_test_doc_ids[1].update(query["answer_pids"])
#             elif test_cnt2 < need_test_query_counts[2]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session2_queries.jsonl"
#                 test_cnt2 += 1
#                 answer_test_doc_ids[2].update(query["answer_pids"])
#             elif test_cnt3 < need_test_query_counts[3]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session3_queries.jsonl"
#                 test_cnt3 += 1
#                 answer_test_doc_ids[3].update(query["answer_pids"])
#             else:
#                 break
#             append_to_jsonl(dest_path, query)
#         print(
#             f"[DONE] Train Query {domain} | {train_cnt0} / {train_cnt1} / {train_cnt2} / {train_cnt3}"
#         )
#         print(
#             f"[DONE] Test Query {domain} | {test_cnt0} / {test_cnt1} / {test_cnt2} / {test_cnt3}"
#         )
#         doc_path = f"/home/work/retrieval/data/raw/{dataset}/{domain}_docs.jsonl"
#         docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
#         train_need_doc_counts = [
#             len(answer_train_doc_ids[0]) / ans_rate * 100,
#             len(answer_train_doc_ids[1]) / ans_rate * 100,
#             len(answer_train_doc_ids[2]) / ans_rate * 100,
#             len(answer_train_doc_ids[3]) / ans_rate * 100,
#         ]
#         test_need_doc_counts = [
#             len(answer_test_doc_ids[0]) / ans_rate * 100,
#             len(answer_test_doc_ids[1]) / ans_rate * 100,
#             len(answer_test_doc_ids[2]) / ans_rate * 100,
#             len(answer_test_doc_ids[3]) / ans_rate * 100,
#         ]
#         print(
#             f"train_need_doc_counts: {train_need_doc_counts}, test_need_doc_counts: {test_need_doc_counts}"
#         )

#         # TRAIN
#         train_cnt0, train_cnt1, train_cnt2, train_cnt3 = 0, 0, 0, 0
#         for doc_id in answer_train_doc_ids[0]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session0_docs.jsonl"
#             train_cnt0 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         for doc_id in answer_train_doc_ids[1]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session1_docs.jsonl"
#             train_cnt1 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         for doc_id in answer_train_doc_ids[2]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session2_docs.jsonl"
#             train_cnt2 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         for doc_id in answer_train_doc_ids[3]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session3_docs.jsonl"
#             train_cnt3 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         print(
#             f"[DONE] Train Document {domain} answer  {train_cnt0} / {train_cnt1} / {train_cnt2} / {train_cnt3}"
#         )
#         # TEST
#         test_cnt0, test_cnt1, test_cnt2, test_cnt3 = 0, 0, 0, 0
#         for doc_id in answer_test_doc_ids[0]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session0_docs.jsonl"
#             test_cnt0 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         for doc_id in answer_test_doc_ids[1]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session1_docs.jsonl"
#             test_cnt1 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         for doc_id in answer_test_doc_ids[2]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session2_docs.jsonl"
#             test_cnt2 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         for doc_id in answer_test_doc_ids[3]:
#             dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session3_docs.jsonl"
#             test_cnt3 += 1
#             append_to_jsonl(dest_path, docs[doc_id])
#         print(
#             f"[DONE] Test Document {domain} answer  {test_cnt0} / {test_cnt1} / {test_cnt2} / {test_cnt3}"
#         )

#         left_doc_ids = (
#             set(docs.keys())
#             - answer_train_doc_ids[0]
#             - answer_train_doc_ids[1]
#             - answer_train_doc_ids[2]
#             - answer_train_doc_ids[3]
#             - answer_test_doc_ids[0]
#             - answer_test_doc_ids[1]
#             - answer_test_doc_ids[2]
#             - answer_test_doc_ids[3]
#         )
#         left_doc_ids = list(left_doc_ids)
#         random.shuffle(left_doc_ids)
#         for doc_cnt, doc_id in enumerate(left_doc_ids):
#             # TRAIN
#             if train_cnt0 < train_need_doc_counts[0]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session0_docs.jsonl"
#                 train_cnt0 += 1
#             elif train_cnt1 < train_need_doc_counts[1]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session1_docs.jsonl"
#                 train_cnt1 += 1
#             elif train_cnt2 < train_need_doc_counts[2]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session2_docs.jsonl"
#                 train_cnt2 += 1
#             elif train_cnt3 < train_need_doc_counts[3]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/train_session3_docs.jsonl"
#                 train_cnt3 += 1
#             # TEST
#             elif test_cnt0 < test_need_doc_counts[0]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session0_docs.jsonl"
#                 test_cnt0 += 1
#             elif test_cnt1 < test_need_doc_counts[1]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session1_docs.jsonl"
#                 test_cnt1 += 1
#             elif test_cnt2 < test_need_doc_counts[2]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session2_docs.jsonl"
#                 test_cnt2 += 1
#             elif test_cnt3 < test_need_doc_counts[3]:
#                 dest_path = f"/home/work/retrieval/data/raw/{dataset}/test_session3_docs.jsonl"
#                 test_cnt3 += 1
#             else:
#                 break
#             append_to_jsonl(dest_path, docs[doc_id])
#             if doc_cnt % 1000 == 0:
#                 print(
#                     f"Document Train {domain} no-answer | {train_cnt0}/{train_cnt1}/{train_cnt2}/{train_cnt3}"
#                 )
#                 print(
#                     f"Document Test {domain} no-answer | {test_cnt0}/{test_cnt1}/{test_cnt2}/{test_cnt3}"
#                 )
#         print(
#             f"[DONE] Document Train {domain} no-answer | {train_cnt0}/{train_cnt1}/{train_cnt2}/{train_cnt3}"
#         )
#         print(
#             f"[DONE] Document Test {domain} no-answer | {test_cnt0}/{test_cnt1}/{test_cnt2}/{test_cnt3}"
#         )


def sessioning(
    domains,
    domain_answer_rate,
    need_train_query_counts,
    need_test_query_counts,
    dataset="lotte",
):
    session_count = len(need_train_query_counts)
    print(f"Session count {session_count}")

    for domain, ans_rate in zip(domains, domain_answer_rate):
        queries = read_jsonl(
            f"/home/work/retrieval/data/raw/{dataset}/{domain}_queries.jsonl", True
        )
        random.shuffle(queries)

        answer_train_doc_ids = defaultdict(set)
        answer_test_doc_ids = defaultdict(set)

        print(
            f"need_train_query_counts: {need_train_query_counts}, need_test_query_counts: {need_test_query_counts}"
        )

        train_cnts = [0] * session_count
        test_cnts = [0] * session_count

        for query in queries:
            assigned = False
            for i in range(session_count):
                if train_cnts[i] < need_train_query_counts[i]:
                    dest_path = f"/home/work/retrieval/data/sub2/{dataset}/train_session{i}_queries.jsonl"
                    train_cnts[i] += 1
                    answer_train_doc_ids[i].update(query["answer_pids"])
                    append_to_jsonl(dest_path, query)
                    assigned = True
                    break
            if not assigned:
                for i in range(session_count):
                    if test_cnts[i] < need_test_query_counts[i]:
                        dest_path = f"/home/work/retrieval/data/sub2/{dataset}/test_session{i}_queries.jsonl"
                        test_cnts[i] += 1
                        answer_test_doc_ids[i].update(query["answer_pids"])
                        append_to_jsonl(dest_path, query)
                        assigned = True
                        break
            if not assigned:
                break  # 모든 쿼리 세션이 채워졌으면 종료

        print(f"[DONE] Train Query {domain} | {' / '.join(map(str, train_cnts))}")
        print(f"[DONE] Test Query {domain} | {' / '.join(map(str, test_cnts))}")

        # 문서 세션화
        doc_path = f"/home/work/retrieval/data/raw/{dataset}/{domain}_docs.jsonl"
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")

        train_need_doc_counts = [
            len(answer_train_doc_ids[i]) / ans_rate * 100 for i in range(session_count)
        ]
        test_need_doc_counts = [
            len(answer_test_doc_ids[i]) / ans_rate * 100 for i in range(session_count)
        ]

        print(
            f"train_need_doc_counts: {train_need_doc_counts}, test_need_doc_counts: {test_need_doc_counts}"
        )

        train_cnts = [0] * session_count
        test_cnts = [0] * session_count

        # 답변 문서 우선 저장
        for i in range(session_count):
            for doc_id in answer_train_doc_ids[i]:
                dest_path = f"/home/work/retrieval/data/sub2/{dataset}/train_session{i}_docs.jsonl"
                train_cnts[i] += 1
                append_to_jsonl(dest_path, docs[doc_id])
            for doc_id in answer_test_doc_ids[i]:
                dest_path = f"/home/work/retrieval/data/sub2/{dataset}/test_session{i}_docs.jsonl"
                test_cnts[i] += 1
                append_to_jsonl(dest_path, docs[doc_id])

        print(
            f"[DONE] Train Document {domain} answer  {' / '.join(map(str, train_cnts))}"
        )
        print(
            f"[DONE] Test Document {domain} answer  {' / '.join(map(str, test_cnts))}"
        )

        # 남은 문서들 계산 (겹치지 않도록)
        used_doc_ids = set()
        for i in range(session_count):
            used_doc_ids.update(answer_train_doc_ids[i])
            used_doc_ids.update(answer_test_doc_ids[i])

        left_doc_ids = list(set(docs.keys()) - used_doc_ids)
        random.shuffle(left_doc_ids)

        for doc_cnt, doc_id in enumerate(left_doc_ids):
            assigned = False
            for i in range(session_count):
                if train_cnts[i] < train_need_doc_counts[i]:
                    dest_path = f"/home/work/retrieval/data/sub2/{dataset}/train_session{i}_docs.jsonl"
                    train_cnts[i] += 1
                    append_to_jsonl(dest_path, docs[doc_id])
                    assigned = True
                    break
            if not assigned:
                for i in range(session_count):
                    if test_cnts[i] < test_need_doc_counts[i]:
                        dest_path = f"/home/work/retrieval/data/sub2/{dataset}/test_session{i}_docs.jsonl"
                        test_cnts[i] += 1
                        append_to_jsonl(dest_path, docs[doc_id])
                        assigned = True
                        break
            if not assigned:
                break
            if doc_cnt % 1000 == 0:
                print(
                    f"Document Train {domain} no-answer | {' / '.join(map(str, train_cnts))}"
                )
                print(
                    f"Document Test {domain} no-answer | {' / '.join(map(str, test_cnts))}"
                )

        print(
            f"[DONE] Document Train {domain} no-answer | {' / '.join(map(str, train_cnts))}"
        )
        print(
            f"[DONE] Document Test {domain} no-answer | {' / '.join(map(str, test_cnts))}"
        )


def split_data(queries, documents, num_splits=3, queries_per_prefix=13):
    doc_dict = {doc["doc_id"]: doc for doc in documents}
    query_prefix_groups = defaultdict(list)

    for query in queries:
        prefix = query["qid"].split("_")[0]
        query_prefix_groups[prefix].append(query)

    query_subsets = [[] for _ in range(num_splits)]
    doc_subsets = [set() for _ in range(num_splits)]

    for prefix, query_list in query_prefix_groups.items():
        random.shuffle(query_list)
        prefix_splits = [
            query_list[i : i + queries_per_prefix]
            for i in range(0, len(query_list), queries_per_prefix)
        ]

        for i, split in enumerate(prefix_splits):
            subset_idx = i % num_splits
            query_subsets[subset_idx].extend(split)
            for query in split:
                doc_subsets[subset_idx].update(query["answer_pids"])
    print(f"Query Subsets Distribution: {[len(q) for q in query_subsets]}")
    doc_subsets = [
        [doc_dict[pid] for pid in doc_set if pid in doc_dict] for doc_set in doc_subsets
    ]

    remaining_docs = set(doc_dict.keys()) - set().union(
        *[set(d["doc_id"] for d in subset) for subset in doc_subsets]
    )
    remaining_docs = list(remaining_docs)
    random.shuffle(remaining_docs)

    for i, doc_id in enumerate(remaining_docs):
        doc_subsets[i % num_splits].append(doc_dict[doc_id])

    doc_subsets = [list(subset) for subset in doc_subsets]

    def print_distribution(subsets, name, _id):
        for i, subset in enumerate(subsets):
            total_count = len(subset)
            if total_count == 0:
                print(f"{name} Subset {i+1} Distribution: Empty")
                continue

            prefix_counts = Counter([item[_id].split("_")[0] for item in subset])
            prefix_ratios = {
                prefix: round((count / total_count) * 100, 2)
                for prefix, count in prefix_counts.items()
            }

            sorted_ratios = {
                key: prefix_ratios[key] for key in sorted(prefix_ratios.keys())
            }

            print(
                f"{name} Subset {i+1} Distribution: {sorted_ratios} (Total: {total_count})"
            )

    print_distribution(query_subsets, "Queries", "qid")
    print_distribution(doc_subsets, "Documents", "doc_id")
    return query_subsets, doc_subsets


def read_jsonl_line(filepath):
    res = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            res.append(json.loads(line.strip()))
    return res


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def process_hotpot_qa(path="../data/hotpot_dev_distractor_v1.json"):
    datas = read_jsonl_line(path)[0]
    processed = []
    for i, data in enumerate(datas):
        question = data["question"]
        golden_answer_parts = [data["answer"]]
        for fact in data["supporting_facts"]:
            context_name, sentence_idx = fact
            for context_item in data["context"]:
                if context_item[0] == context_name:
                    if sentence_idx < len(context_item[1]):
                        golden_answer_parts.append(context_item[1][sentence_idx])
                    else:
                        print(
                            f"Warning: Index {sentence_idx} out of range for context '{context_name}'"
                        )
                        continue
        golden_answer = " ".join(golden_answer_parts)
        qna = {"query": question, "answer": golden_answer}
        processed.append(qna)
        if i % 1000 == 0:
            print(f"{i}th query and answer: {qna}")
        if i == 15000:
            break
    save_jsonl(processed, "/home/work/retrieval/data/raw/lotte/pretrained,jsonl")


def check_duplicate(session_count=12):
    doc_id_to_files = defaultdict(set)
    for i in range(session_count):
        src_docs_path = (
            f"/home/work/retrieval/data/raw/lotte/train_session{i}_docs.jsonl"
        )
        with open(src_docs_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj.get("doc_id")
                if doc_id is not None:
                    doc_id_to_files[doc_id].add(src_docs_path)
    duplicateds = {
        doc_id: sorted(list(files))
        for doc_id, files in doc_id_to_files.items()
        if len(files) > 1
    }
    for doc_id, files in list(duplicateds.items()):
        print(f"{doc_id} | {files}")


if __name__ == "__main__":
    sessioning(
        dataset="lotte",
        domains=[
            "technology",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[1.82],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[
            0,
            300,
            0,
            0,
            0,
            0,
            300,
            0,
            0,
            0,
        ],  # [150, 150, 0, 0, 0, 0, 150, 150, 0]
        need_test_query_counts=[
            0,
            50,
            50,
            0,
            0,
            0,
            50,
            50,
            0,
            0,
        ],  # [50, 50, 0, 0, 0, 0, 50, 50, 0], #
    )
    sessioning(
        dataset="lotte",
        domains=[
            "writing",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[6.47],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[300, 0, 0, 0, 0, 0, 0, 0, 300, 0],
        need_test_query_counts=[
            50,
            50,
            0,
            0,
            0,
            0,
            0,
            0,
            50,
            50,
        ],  # [50, 0, 0, 0, 50, 50, 0, 0, 50],#
    )

    # sessioning(
    #     dataset="lotte",
    #     domains=[
    #         "recreation",
    #     ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
    #     domain_answer_rate=[6.43],  # 1.82, 6.47, 1.47, 5.17, 6.43
    #     need_train_query_counts=[0,0,300, 0,0,300, 0,0,0, 0], #[0, 150, 150, 0, 0, 150, 150,  0, 0], #
    #     need_test_query_counts=[0,0,50, 50,0,50, 50,0,0, 0] #[0, 50, 50, 0,  0, 50, 50, 0, 0], #
    # )

    # sessioning(
    #     dataset="lotte",
    #     domains=[
    #         "lifestyle",
    #     ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
    #     domain_answer_rate=[6.43],  # 1.82, 6.47, 1.47, 5.17, 6.43
    #     need_train_query_counts=[0,0,0, 300,0,0, 0,0,0, 300], #[0, 0, 150, 150, 0, 0, 0, 0, 0],
    #     need_test_query_counts=[0,0,0, 50,50,0, 0,0,0, 50] #[0, 0, 50, 50, 0, 0, 0, 0, 0],
    # )
    sessioning(
        dataset="lotte",
        domains=[
            "science",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[1.47],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[0, 0, 0, 0, 300, 0, 0, 300, 0, 0],
        need_test_query_counts=[0, 0, 0, 0, 50, 50, 0, 50, 50, 0],
    )

    # cnt = 3
    # for i in range(1, 4):
    #     src_queries_path = f"/home/work/retrieval/data/raw/lotte/test_session{i}_queries.jsonl"
    #     src_docs_path = f"/home/work/retrieval/data/raw/lotte/test_session{i}_docs.jsonl"
    #     with open(src_queries_path, "r", encoding="utf-8") as f:
    #         queries = [json.loads(line) for line in f]
    #     with open(src_docs_path, "r", encoding="utf-8") as f:
    #         documents = [json.loads(line) for line in f]
    #     query_subsets, doc_subsets = split_data(queries, documents, 3, 11)
    #     for query_subset, doc_subset in zip(query_subsets, doc_subsets):
    #         save_jsonl(
    #             query_subset,
    #             f"../data/sub2/lotte/test_session{cnt}_queries.jsonl",
    #         )
    #         save_jsonl(
    #             doc_subset,
    #             f"../data/sub2/lotte/test_session{cnt}_docs.jsonl",
    #         )
    #         cnt += 1
