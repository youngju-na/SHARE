# Train (DTU)
# src/dataset/__init__.py에서 Re10k 주석처리 
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/home/youngju/ssd/ufosplat/rebuttal-ablation/ours \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true

# Test (DTU)
# checkpoint 경로 확인!
python -m src.main \
    wandb.mode=disabled \
    wandb.project=ReconGS \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufosplat/rebuttal/results/ours-3views-23_24-22_33 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.make_baseline_1=true \
    dataset.baseline_scale_bounds=true \
    dataset.padding_size=0 \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/home/youngju/ssd/ufosplat/rebuttal/pose-normalized-dtu/epoch_40-step_140000.ckpt

python -m src.main \
    wandb.mode=disabled \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufosplat/rebuttal/results/ours-2views-23_24_33/ \
    test.output_path=/home/youngju/ssd/ufosplat/rebuttal/results/ours-2views-23_24_33 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=2 \
    dataset.test_context_views=[23,24] \
    dataset.test_target_views=[33] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.padding_size=0 \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=3 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/home/youngju/ssd/ufosplat/rebuttal/pose-normalized-dtu/epoch_40-step_140000.ckpt