# Train (DTU) -> Test (Re10K)
# checkpoint 경로 확인!
python -m src.main \
    wandb.mode=disabled \
    wandb.project=ReconGS \
    +experiment=re10k \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufosplat/rebuttal/results/cross-dataset-dtu2re10k \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=2 \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.padding_size=0 \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/home/youngju/ssd/ufosplat/checkpoints/epoch_28-step_100000.ckpt


# Train (Re10k) -> Test (DTU)
# checkpoint 경로 확인!
python -m src.main \
    wandb.mode=disabled \
    wandb.project=ReconGS \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufosplat/rebuttal/results/cross-dataset-re10k2dtu \
    test.output_path=/home/youngju/ssd/ufosplat/rebuttal/results/cross-dataset-re10k2dtu \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=2 \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[256,256] \
    dataset.test_context_views=[23,34] \
    dataset.test_target_views=[22,15,34] \
    dataset.baseline_scale_bounds=true \
    dataset.make_baseline_1=true \
    dataset.padding_size=0 \
    model.encoder.ray.width=64 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/home/youngju/ssd/ufosplat/rebuttal/re10k-pose-normalized/epoch_4-step_300000.ckpt

