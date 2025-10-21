#!/bin/bash

# Fixed reference views
fixed_views=(23 24)

# List of other views to pair with the fixed reference views
views=(22 15 34 14 32 16 35 25)

# Base command parts that do not change
base_command="python -m src.main \
  +experiment=dtu \
  data_loader.train.num_workers=0 \
  data_loader.test.num_workers=0 \
  data_loader.val.num_workers=0 \
  checkpointing.load=/home/youngju/ssd/ufosplat/checkpoints/reconGS-any-views/recongs-any-views-mvsplat-residual-step_240000.ckpt \
  mode=test \
  model.encoder.pred_campose=true \
  +experiment.dataset.image_shape=[224,224] \
  dataset/view_sampler=evaluation \
  test.compute_scores=true \
  dataset.view_sampler.num_context_views=3 \
  dataset.make_baseline_1=false \
  dataset.use_test_ref_views_as_src=true \
  dataset.view_selection_type=best \
  dataset.single_view=true \
  model.encoder.predict_only_canonical=true \
  model.encoder.predict_offset=true \
  model.encoder.predict_anchor_gaussian=false \
  model.encoder.rendering_units=[all,anchor,offset] \
  model.encoder.backbone=mvsplat \
  model.encoder.estimation_space=depthmap \
  model.encoder.gaussian_adapter.gaussian_2d_scale=true \
  model.encoder.gaussian_adapter.learn_sh_residual_from_canoncal_rgb=true"

# Loop over each view to pair with the fixed views
for view in "${views[@]}"; do
    if [[ $view -ne ${fixed_views[0]} && $view -ne ${fixed_views[1]} ]]; then
        # Set the specific view combination and output path for this iteration
        test_ref_views="dataset.test_ref_views=[${fixed_views[0]},${fixed_views[1]},$view]"
        output_path="test.output_path=test/reconGS-any-views-3/${fixed_views[0]}-${fixed_views[1]}-$view"

        # Construct and run the full command
        full_command="$base_command $test_ref_views $output_path"
        echo "Running command with views ${fixed_views[0]}, ${fixed_views[1]}, and $view"
        eval $full_command
    fi
done