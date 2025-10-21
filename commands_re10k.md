# Train (re10k)
# src/dataset/__init__.py => choose which dataset to use here (comment out DTU)
python -m src.main \
    wandb.mode=online \
    +experiment=re10k \
    wandb.project=SHARE-rebuttal-re10k \
    output_dir=/mnt/nas4/youngju/ufosplat/rebuttal/re10k-pose-improve-2 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=2 \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[256,256] \
    dataset.padding_size=0 \
    dataset.make_baseline_1=true \
    model.encoder.ray.width=64 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true

# Test (re10k) 
# checkpoint path
python -m src.main \
    wandb.mode=online \
    +experiment=re10k \
    dataset/view_sampler=evaluation \
    mode=test \
    test.compute_scores=true \
    wandb.project=SHARE-rebuttal-re10k \
    test.output_path=rebuttal/re10k-pose-normalized/results \
    output_dir=/mnt/nas4/youngju/ufosplat/rebuttal/re10k-pose-normalized/results \
    data_loader.train.batch_size=1 \
    dataset.baseline_scale_bounds=true \
    dataset.make_baseline_1=true \
    dataset.view_sampler.num_context_views=2 \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[256,256] \
    dataset.padding_size=0 \
    model.encoder.ray.width=64 \
    model.encoder.ray.depth=4 \
    model.encoder.ray.mlp_ratio=2.0 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/home/youngju/ssd/ufosplat/rebuttal/re10k-pose-normalized/epoch_4-step_300000.ckpt
