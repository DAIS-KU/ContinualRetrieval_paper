import collections
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .buffer_utils import (
    cosine_similarity,
    cosine_similarity_3d,
    get_grad_vector,
    random_retrieve,
)


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/buffer/our_retrieve_emb_cosine.py#L9
class L2R_retrieve(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.train_params = train_params
        self.alpha = params.alpha
        self.beta = params.beta
        self.new_bz = params.new_batch_size
        """
            https://www.notion.so/02-21-Baseline-Dataset-1a0231d8f10a808aa069c8e13e0cc2e5
            - mem_upsample: 메모리에서 샘플링할 최대 문서 수로, 코드 보시면 버퍼에서 mem_upsample 만큼 random_retrieve 합니다!
            - mem_bz:  training시 이용할 메모리 데이터 수.
            - ocs에서도 training시 메모리에서 random sampling한 B_c 이용해 training. 만약 메모리에서 샘플링하는 수랑, 실제 training에서 이용할 메모리 데이터 수가 일치하면 통일.    
        """
        self.mem_upsample = params.mem_upsample
        self.mem_bz = params.mem_batch_size

    def retrieve(self, buffer, qid_lst, docids_lst, **kwargs):
        model_temp = copy.deepcopy(buffer.model)
        model_temp.eval()

        batch_size = len(qid_lst)
        n_doc = len(docids_lst) // len(qid_lst)
        docids_pos_lst = np.array(docids_lst).reshape(batch_size, n_doc)[
            :, :1
        ]  # pos passage
        docids_neg_lst = np.array(docids_lst).reshape(batch_size, n_doc)[
            :, 1:
        ]  # 去掉pos passage
        q_lst, d_lst = kwargs["q_lst"], kwargs["d_lst"]

        res_d_lst = collections.defaultdict(list)
        res_neg_did_lst = collections.defaultdict(set)

        ############## 处理new data #############
        if self.params.compatible:  # 正例用 old embedding
            identity = torch.arange(len(qid_lst)) * n_doc
            doc_oldemb = []
            for docid in docids_pos_lst[:, 0]:
                doc_oldemb.append(buffer.buffer_did2emb[int(docid)])
            doc_oldemb = torch.tensor(np.array(doc_oldemb)).to(
                self.train_params.device
            )  # [num_q, 768]
            new_model_out = model_temp(q_lst, d_lst, identity, doc_oldemb)
        else:
            new_model_out = model_temp(q_lst, d_lst)
        index_new = self.get_new_data(
            new_model_out, self.new_bz, self.alpha, self.beta
        )  # 得到选择出来的新数据的负例下标[batch_size, new_bz], cuda
        for i, qid in enumerate(qid_lst):
            res_neg_did_lst[qid].update(
                docids_neg_lst[i][index_new[i].cpu()]
            )  # 选择出来的负例
        index_new = torch.cat(
            (torch.zeros_like(index_new[:, :1]), index_new + 1), dim=-1
        )  # [batch_size, new_bz+1]
        for key, val in d_lst.items():
            val = val.reshape(batch_size, -1, val.size(-1))
            res_d_lst[key].append(
                torch.gather(
                    val, 1, index_new.unsqueeze(dim=2).repeat(1, 1, val.size(-1))
                )
            )  # [batch_size, new_bz+1, 128]

        ############### 处理mem data ##############
        buffer_len = min([len(buffer.buffer_qid2dids[qid]) for qid in qid_lst])
        mem_upsample = min(self.mem_upsample, buffer_len)
        mem_bz = min(mem_upsample, self.mem_bz)
        # print(f"buffer_len:{buffer_len}, mem_upsample:{mem_upsample}/{self.mem_upsample}, mem_bz: {mem_bz}/{self.mem_bz}")
        if mem_upsample > 0 and mem_bz > 0:
            mem_upsample_docids_lst = []
            for i, qid in enumerate(qid_lst):
                mem_upsample_docids = random_retrieve(
                    buffer.buffer_qid2dids[qid], mem_upsample
                )
                mem_upsample_docids_lst.extend(
                    docids_pos_lst[i].tolist() + mem_upsample_docids
                )  # 把正例也加进去
            print(f"mem_upsample_docids_lst: {mem_upsample_docids_lst}")
            mem_doc_lst = [buffer.did2doc[did] for did in mem_upsample_docids_lst]
            mem_doc_lst = buffer.tokenizer.batch_encode_plus(
                mem_doc_lst,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.params.p_max_len,
                truncation="only_first",
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            if self.params.compatible:  # 全部用old embedding，不需要再过模型
                identity = torch.arange(len(mem_upsample_docids_lst))
                doc_oldemb = []
                for docid in mem_upsample_docids_lst:
                    doc_oldemb.append(buffer.buffer_did2emb[int(docid)])
                doc_oldemb = torch.tensor(np.array(doc_oldemb)).to(
                    self.train_params.device
                )  # .to('cuda:0')  # [num_q*(1+mem_upsample), 768]
                index_mem = self.get_mem_data(
                    new_model_out, index_new, doc_oldemb, mem_bz
                )  # [batch_size, mem_bz]
            else:
                for key, value in mem_doc_lst.items():
                    mem_doc_lst[key] = value.to(
                        "cuda:0"
                    )  # mem_doc_lst = [batch_size*(1+mem_upsample), 128]
                mem_q_lst = {}
                for key, value in q_lst.items():
                    mem_q_lst[key] = q_lst[key].clone().to("cuda:0")
                model_temp = model_temp.to("cuda:0")
                mem_model_out = model_temp(mem_q_lst, mem_doc_lst)
                mem_p_reps = mem_model_out.p_reps  # [bz*n, 768]
                index_mem = self.get_mem_data(
                    new_model_out, index_new, mem_p_reps, mem_bz
                )  # [batch_size, mem_bz]
            index_mem = (
                index_mem.to(self.train_params.device) + 1
            )  # [batch_size, mem_bz]
            for key, val in mem_doc_lst.items():
                val = val.to(self.train_params.device).reshape(
                    batch_size, -1, val.size(-1)
                )  # # [batch_size, mem_upsample, 128]
                res_d_lst[key].append(
                    torch.gather(
                        val, 1, index_mem.unsqueeze(dim=2).repeat(1, 1, val.size(-1))
                    )
                )  # [batch_size, mem_bz, 128]

        for key, val in res_d_lst.items():
            val = torch.cat(val, dim=1)  # [batch_size, new_bz+mem_bz+1, 128]
            res_d_lst[key] = val.reshape(
                -1, val.size(-1)
            )  # [batch_size*(new_bz+mem_bz+1), 128]
        if self.params.compatible:
            doc_oldemb = doc_oldemb.reshape(
                batch_size, -1, doc_oldemb.size(-1)
            )  # # [batch_size, 1+ mem_upsample, 768]
            doc_oldemb = torch.cat(
                (
                    doc_oldemb[:, :1, :],
                    torch.gather(
                        doc_oldemb,
                        1,
                        index_mem.unsqueeze(dim=2).repeat(1, 1, doc_oldemb.size(-1)),
                    ),
                ),
                dim=1,
            )  # # [batch_size, 1+ mem_bz, 768]
            doc_oldemb = doc_oldemb.reshape(
                -1, doc_oldemb.size(-1)
            )  # [batch_size*(1+mem_bz), 768]
            return doc_oldemb, res_d_lst, None, res_neg_did_lst
        return res_d_lst, None, res_neg_did_lst

    def get_mem_data(self, new_model_out, index_new, mem_p_reps, mem_bz):
        new_q_reps = new_model_out.q_reps
        new_p_reps = new_model_out.p_reps
        new_p_reps = new_p_reps.reshape(
            new_q_reps.size(0), -1, new_p_reps.size(1)
        )  # [bz, n, 768]
        choiced_new_reps = torch.gather(
            new_p_reps, 1, index_new.unsqueeze(dim=2).repeat(1, 1, new_p_reps.size(-1))
        )[
            :, 1:, :
        ]  # [batch_size, new_bz, 768]
        if not self.params.compatible:
            choiced_new_reps = choiced_new_reps.to("cuda:0")

        p_reps = mem_p_reps  # mem_model_out.p_reps  # [bz*n, 768]
        p_reps = p_reps.reshape(new_q_reps.size(0), -1, p_reps.size(1))  # [bz, n, 768]
        neg_p_reps = p_reps[:, 1:, :]  # [bz, n-1, 768]

        inter_sim = cosine_similarity_3d(
            neg_p_reps, choiced_new_reps
        )  # [bz, n-1, new_bz]
        inter_sim_sum = torch.sum(inter_sim, dim=-1)  # [bz, n-1]
        inter_sim = inter_sim_sum * (-1.0) / inter_sim.size(-1)  # [bz, n-1], 相似度尽可能小

        indexs = inter_sim.sort(dim=1, descending=True)[1][:, :mem_bz]
        return indexs

    def get_new_data(self, new_model_out, new_bz, alpha, beta):
        q_reps = new_model_out.q_reps  # [8, 768]
        # print("q_reps: ", q_reps.shape)
        p_reps = new_model_out.p_reps  # [8*n, 768]
        # print("p_reps: ", p_reps.shape)
        p_reps = p_reps.reshape(q_reps.size(0), -1, p_reps.size(1))  # [8, n, 768]

        q_reps_norm = q_reps.norm(p=2, dim=1, keepdim=True)  # [8, 1]
        q_reps = q_reps.unsqueeze(dim=1)  # [8, 1, 768]
        p_q = (
            torch.matmul(p_reps, q_reps.transpose(1, 2))
            * q_reps
            / (q_reps_norm * q_reps_norm).unsqueeze(dim=2).clamp(min=1e-8)
        )  # p在q方向上的投影向量, [8, n, 768]
        neg = p_q[:, 1:, :]  # [8, n-1, 768]
        pos = p_q[:, :1, :].repeat(1, neg.size(1), 1)  # [8, n-1, 768]
        dis = F.pairwise_distance(
            neg.reshape(-1, neg.size(-1)), pos.reshape(-1, pos.size(-1)), p=2.0
        ).reshape(
            neg.size(0), -1
        )  # [8, n-1], 尽可能大
        # print(f"p_q:{p_q.shape}, pos:{pos.shape}, neg:{neg.shape}")
        # pos:tensor([], size=(1, 0, 768), grad_fn=<RepeatBackward0>), neg:tensor([], size=(1, 0, 768), grad_fn=<SliceBackward0>)

        neg_p_reps = p_reps[:, 1:, :]  # [8, n-1, 768]
        inter_sim = cosine_similarity_3d(neg_p_reps, neg_p_reps)  # [8, n-1, n-1]
        inter_sim_sum = torch.sum(inter_sim, dim=-1)  # [8, n-1]
        inter_sim = (inter_sim_sum - torch.ones_like(inter_sim_sum)) / (
            inter_sim.size(-1) - 1
        )  # [8, n-1], 尽可能小
        # print(
        #     f"neg_p_reps:{neg_p_reps.shape}, inter_sim:{inter_sim.shape}, inter_sim_sum:{inter_sim_sum.shape}"
        # )

        # norm dis
        mean_dis = torch.mean(dis, dim=-1, keepdim=True)  # [8, 1]
        std_dis = torch.std(dis, dim=-1, keepdim=True)  # [8, 1]
        dis = (dis - mean_dis) / std_dis

        # norm inter_sim
        mean_inter_sim = torch.mean(inter_sim, dim=-1, keepdim=True)  # [8, 1]
        std_inter_sim = torch.std(inter_sim, dim=-1, keepdim=True)  # [8, 1]
        inter_sim = (inter_sim - mean_inter_sim) / std_inter_sim

        sim = alpha * dis - beta * inter_sim
        indexs = sim.sort(dim=1, descending=True)[1][:, :new_bz]
        return indexs
