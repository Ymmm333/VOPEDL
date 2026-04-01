
if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed)
    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader: dict = getDataLoader(args, source_info, target_info, drop_last=True)

    model = get_model(args, source_info, target_info)

    dataset_key = f"{args.source_dataset}-{args.target_dataset}"
    trainer = Trainer(model, model.device, ckpt_dir=f"checkpoints/{dataset_key}")

    if args.pre_train == 'True':
        trainer.train('pre_train', data_loader['source']['train'], args.pre_train_epochs,
                      resume=args.resume, seed=args.seed)

    trainer.train(
        'train',
        CombinedLoader([data_loader['source']['train'], data_loader['target']['train']]),
        args.epochs,
        resume=args.resume,
        seed=args.seed
    )

    trainer.test('test', data_loader['target']['test'], seed=args.seed)

    trainer.print_history_summary()

    if model.args.draw == 'True':
        trainer.test('prediction', data_loader['target']['all'])



