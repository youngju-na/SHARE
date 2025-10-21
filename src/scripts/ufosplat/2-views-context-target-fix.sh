#!/bin/bash

# Fixed views
context_views=(23 24 25)
target_views=(23 24 25)

# Base command
base_command="python -m src.main \
  +experiment=dtu \
  checkpointing.load=/home/youngju/ssd/ufosplat/checkpoints/SHARE/epoch_9-step_240000.ckpt \
  mode=test \
  model.encoder.pred_campose=true \
  +experiment.dataset.image_shape=[224,224] \
  dataset/view_sampler=evaluation \
  test.compute_scores=true \
  dataset.view_sampler.num_context_views=3 \
  dataset.make_baseline_1=false \
  dataset.view_selection_type=best \
  dataset.single_view=true \
  model.encoder.predict_only_canonical=true \
  model.encoder.predict_offset=true \
  model.encoder.predict_anchor_gaussian=false \
  model.encoder.rendering_units=[all,anchor,offset] \
  model.encoder.backbone=mvsplat \
  model.encoder.estimation_space=depthmap \
  model.encoder.gaussian_adapter.gaussian_2d_scale=true \
  model.encoder.gaussian_adapter.learn_sh_residual_from_canoncal_rgb=true \
  model.encoder.use_monoN_loss=true \
  model.encoder.lifting.lifting_switch=true"

# Function to join array elements
join_by() {
    local d=${1-} f=${2-}
    if shift 2; then
        printf %s "$f" "${@/#/$d}"
    fi
}

# Construct view strings
context_view_str=$(join_by ',' "${context_views[@]}")
target_view_str=$(join_by ',' "${target_views[@]}")

# Set view parameters
view_params="dataset.test_context_views=[$context_view_str] \
             dataset.test_target_views=[$target_view_str]"

# Construct output path (using underscores instead of commas)
output_path_str=$(join_by '-' "${context_views[@]}")_$(join_by '-' "${target_views[@]}")
output_path="test.output_path=test/reconGS-any-views/$output_path_str"

# Construct and run the full command
full_command="$base_command $view_params $output_path"
echo "Running command with context views [$context_view_str] and target views [$target_view_str]"
echo "Output path: $output_path"
eval $full_command