

# Ours
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/mnt/nas4/youngju/rebuttal-ablation/wo-mean-var \
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
    model.encoder.feature_fusion_strategy=weighted_sum \
    model.encoder.ray.pose_embedding=true

# w/o mean variance volume
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/mnt/nas4/youngju/rebuttal-ablation/wo-mean-var \
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
    model.encoder.feature_fusion_strategy=weighted_sum \
    model.encoder.ray.pose_embedding=true

# w/ offset 2
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/mnt/nas4/youngju/rebuttal-ablation/w-offset-2 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=2 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true

# w/ offset 1
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/mnt/nas4/youngju/rebuttal-ablation/w-offset-1 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=1 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true

# anchor only
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/mnt/nas4/youngju/rebuttal-ablation/anchor-only \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=0 \
    model.encoder.predict_offset=false \
    model.encoder.predict_anchor_gaussian=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true

# w/o pose embedding
python -m src.main \
    wandb.mode=online \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    output_dir=/mnt/nas4/youngju/rebuttal-ablation/wo-pose-embedding \
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
    model.encoder.ray.pose_embedding=false


# ------------------------------------Test ------------------------------------
# ------------------------------------Test ------------------------------------
# ------------------------------------Test ------------------------------------
# ------------------------------------Test ------------------------------------

# w/o mean variance volume
python -m src.main \
    wandb.mode=disabled \
    wandb.project=SHARE-rebuttal \
    mode=test \
    test.compute_scores=true \
    +experiment=dtu \
    output_dir=/home/youngju/ssd/ufo_outputs/rebuttal-ablation/wo-mean-var \
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
    model.encoder.feature_fusion_strategy=weighted_sum \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/mnt/nas4/youngju/rebuttal-ablation/wo-mean-var/checkpoints/epoch_40-step_140000.ckpt

# w/ offset 2
python -m src.main \
    wandb.mode=disabled \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufo_outputs/rebuttal-ablation/w-offset-2 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=2 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/mnt/nas4/youngju/rebuttal-ablation/w-offset-2/checkpoints/epoch_40-step_140000.ckpt

# w/ offset 1
python -m src.main \
    wandb.mode=disabled \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufo_outputs/rebuttal-ablation/w-offset-1 \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=1 \
    model.encoder.predict_offset=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/mnt/nas4/youngju/rebuttal-ablation/w-offset-1/checkpoints/epoch_40-step_140000.ckpt

# anchor only
python -m src.main \
    wandb.mode=disabled \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufo_outputs/rebuttal-ablation/anchor-only \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.test_context_views=[23,24,33] \
    dataset.test_target_views=[22,15,34] \
    trainer.num_sanity_val_steps=1 \
    dataset.image_shape=[224,224] \
    dataset.baseline_scale_bounds=true \
    model.encoder.ray.width=56 \
    model.encoder.num_offset_gaussian=0 \
    model.encoder.predict_offset=false \
    model.encoder.predict_anchor_gaussian=true \
    model.encoder.backbone=mvsplat+mast3r \
    model.encoder.feature_fusion_strategy=both \
    model.encoder.ray.pose_embedding=true \
    checkpointing.load=/mnt/nas4/youngju/rebuttal-ablation/anchor-only/checkpoints/epoch_40-step_140000.ckpt

# w/o pose embedding
python -m src.main \
    wandb.mode=disabled \
    wandb.project=SHARE-rebuttal \
    +experiment=dtu \
    mode=test \
    test.compute_scores=true \
    output_dir=/home/youngju/ssd/ufo_outputs/rebuttal-ablation/wo-pose-embedding \
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
    model.encoder.ray.pose_embedding=false \
    checkpointing.load=/mnt/nas4/youngju/rebuttal-ablation/wo-pose-embedding/checkpoints/epoch_40-step_140000.ckpt