import argparse

from pipeline import (
    er_evaluate,
    er_train,
    find_best_k_experiment,
    l2r_evaluate,
    l2r_train,
    mir_evaluate,
    mir_train,
    ocs_evaluate,
    ocs_train,
    proposal_train,
    test_buffer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument(
        "--exp",
        type=str,
        help="experiments(proposal/gt/rand/bm25/find_k/test_buffer/er)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )
    # buffer option
    parser.add_argument(
        "--new_bz",
        default=3,
        help="sampling batch size of new documents.",
    )
    parser.add_argument(
        "--mem_upsample",
        default=6,
        help="candidates sampling from memory.",
    )
    parser.add_argument(
        "--mem_bz",
        default=3,
        help="sampling batch size of old documents from memory.",
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="when it is, compatible",
    )
    # proposal option
    parser.add_argument(
        "--use_label",
        action="store_true",
        help="when it is, use labeled positives. or use sampled positives.",
    )
    parser.add_argument(
        "--use_weight",
        action="store_true",
        help="apply time-decayed weight.",
    )
    parser.add_argument(
        "--use_tensor_key",
        action="store_true",
        help="Use hash as tensor instead of map.",
    )
    parser.add_argument(
        "--negative_k",
        default=6,
        type=int,
        help="number of negative samples.",
    )
    parser.add_argument(
        "--mi",
        default=3,
        type=int,
        help="max iters to converge centroids.",
    )
    parser.add_argument(
        "--wr",
        default=0.2,
        type=float,
        help="warming up rate",
    )
    parser.add_argument(
        "--init_k",
        type=int,
        help="warming up k cluster",
    )
    parser.add_argument(
        "--cmnsz",
        type=int,
        help="cluster minimum number of instances before adding new centroids.",
    )
    parser.add_argument(
        "--sr",
        type=float,
        help="Document stream sampling rate",
    )
    args = parser.parse_args()

    print(f"Running experiments: {args.exp}")
    if args.exp == "proposal":
        print(f"Use label: {args.use_label}")
        print(f"Use weight: {args.use_weight}")
        print(f"Use tensor key: {args.use_tensor_key}")
        print(f"Use max_iters: {args.mi}")
        print(f"Use warmingup_rate: {args.wr}")
        print(f"Use warmingup_k: {args.init_k}")
        print(f"Use cluster_min_size: {args.cmnsz}")
        print(f"Use stream sampling rate: {args.sr}")
        # print(f"Number of Epochs: {args.num_epochs}")
        proposal_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            negative_k=args.negative_k,
            use_label=args.use_label,
            use_weight=args.use_weight,
            use_tensor_key=args.use_tensor_key,
            max_iters=args.mi,
            warmingup_rate=args.wr,
            k=args.init_k,
            cluster_min_size=args.cmnsz,
            sampling_rate=args.sr,
        )
    elif args.exp == "bm25":
        bm25_evaluate()
    elif args.exp == "find_k":
        find_best_k_experiment(max_iters=args.mi, warmingup_rate=args.wr)
    elif args.exp == "er":
        er_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
        )
        er_evaluate()
    elif args.exp == "mir":
        mir_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
        )
        mir_evaluate()
    elif args.exp == "ocs":
        ocs_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
            mem_upsample=args.mem_upsample,
        )
        ocs_evaluate()
    elif args.exp == "l2r":
        l2r_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
            mem_upsample=args.mem_upsample,
        )
        l2r_evaluate()
    else:
        raise ValueError(f"Unsupported experiments {args.exp}")
