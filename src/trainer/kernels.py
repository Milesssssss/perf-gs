"""Data-driven compute-kernel loading for the Pipeline.

Replaces the ~40 hand-written ``self.k_x = mk(dir / "file.slang", "entry")``
lines in ``Pipeline.__init__`` with a single declarative table. To add a
kernel, add one row to ``KERNEL_SPECS``; ``load_kernels`` sets the matching
``target.k_<attr>`` attribute so the ~50 dispatch call sites keep using
``self.k_<name>`` unchanged.

Each spec is ``(attr, relative_path, entry)``:
- ``attr``    — the ``k_<attr>`` attribute name set on the target.
- ``relative_path`` — shader file relative to ``SHADERS_DIR``.
- ``entry``   — the ``[shader("compute")]`` entry point in that file, also used
                as the debug-marker label.
"""

from __future__ import annotations

import slangpy as spy

from trainer.device import SHADERS_DIR


# (attr, relative shader path, entry point). Grouped by shader subdirectory,
# matching the original load order.
KERNEL_SPECS: tuple[tuple[str, str, str], ...] = (
    # rasterize (cluster-cull runs before tile-binning)
    ("build_cluster_aabb",   "rasterize/cluster_cull.slang",                   "buildClusterAABB"),
    ("cull_clusters_frustum", "rasterize/cluster_cull.slang",                  "cullClustersFrustum"),
    # rasterize/tile_binned
    ("clear_textures",       "rasterize/tile_binned/clear.slang",              "clearTextures"),
    ("bin_count",            "rasterize/tile_binned/bin_count.slang",          "binCount"),
    ("bin_emit",             "rasterize/tile_binned/bin_emit.slang",           "binEmit"),
    ("clear_tile_ranges",    "rasterize/tile_binned/tile_ranges.slang",        "clearTileRanges"),
    ("identify_tile_ranges", "rasterize/tile_binned/tile_ranges.slang",        "identifyTileRanges"),
    ("bucket_count",         "rasterize/tile_binned/bucket_count.slang",       "bucketCount"),
    ("forward",              "rasterize/tile_binned/forward.slang",            "forwardRasterize"),
    ("backward",             "rasterize/tile_binned/backward.slang",           "backwardRasterize"),
    ("backward_persplat",    "rasterize/tile_binned/backward_persplat.slang",  "backwardRasterizePerSplat"),
    ("backward_persbucket",  "rasterize/tile_binned/backward_persplat.slang",  "backwardRasterizePerBucket"),
    ("tile_max_processed",   "rasterize/tile_binned/live_buckets.slang",       "tileMaxProcessed"),
    ("scatter_live_buckets", "rasterize/tile_binned/live_buckets.slang",       "scatterLiveBuckets"),
    ("bucket_prepare_dispatch", "rasterize/tile_binned/live_buckets.slang",    "prepareBucketDispatchArgs"),
    ("projection_backward",  "rasterize/tile_binned/projection_backward.slang", "projectionBackward"),
    # losses
    ("loss",                 "losses/l1_l2_loss.slang",                        "computeLossGrad"),
    ("ssim_maps",            "losses/ssim_loss.slang",                         "ssimComputeMaps"),
    ("ssim_grad",            "losses/ssim_loss.slang",                         "ssimGradient"),
    ("regularization",       "losses/regularization.slang",                    "addRegularization"),
    # sort
    ("radix_init_global_hist", "sort/device_radix_sort.slang",                 "initGlobalHist"),
    ("radix_prepare_dispatch", "sort/device_radix_sort.slang",                 "prepareDispatchArgs"),
    ("radix_upsweep",        "sort/device_radix_sort.slang",                   "upsweep"),
    ("radix_scan",           "sort/device_radix_sort.slang",                   "scan"),
    ("radix_downsweep",      "sort/device_radix_sort.slang",                   "downsweep"),
    # optimizers
    ("adam",                 "optimizers/adam.slang",                          "adamStep"),
    ("grad_clear",           "optimizers/grad_utils.slang",                    "clearGradBuffer"),
    ("clear_adam_state",     "optimizers/grad_utils.slang",                    "clearAdamState"),
    # densify
    ("clear_tracking",       "densify/clear_tracking.slang",                   "clearTracking"),
    ("accumulate_iter",      "densify/accumulate_iter.slang",                  "accumulateIter"),
    ("decide_actions_simple", "densify/decide_actions_simple.slang",           "decideActionsSimple"),
    ("split_and_clone",      "densify/split_and_clone.slang",                  "splitAndClone"),
    ("build_prune_mask",     "densify/build_prune_mask.slang",                 "buildPruneMask"),
    ("scatter_compact",      "densify/scatter_compact.slang",                  "scatterCompact"),
    ("scatter_compact_eta",  "densify/scatter_compact.slang",                  "scatterCompactEta"),
    ("opacity_reset",        "densify/opacity_reset.slang",                    "resetOpacity"),
    ("densify_snapshot",     "densify/densify_dispatch.slang",                 "snapshotCount"),
    ("densify_prepare_args", "densify/densify_dispatch.slang",                 "prepareDensifyArgs"),
    # structure-aware densification (SAD-GS)
    ("score_structure_aware", "densify/score_structure_aware.slang",           "scoreStructureAware"),
    ("clear_structure_scores", "densify/score_structure_aware.slang",          "clearStructureScores"),
    ("split_anisotropic",    "densify/split_anisotropic.slang",                "splitAnisotropic"),
    # periodic Morton-order (Z-curve) reorder of the gaussian buffer
    ("morton_encode",        "densify/morton_sort.slang",                      "mortonEncode"),
    ("morton_scatter",       "densify/morton_sort.slang",                      "mortonScatter"),
    ("morton_scatter_eta",   "densify/morton_sort.slang",                      "mortonScatterEta"),
    # wait-free structure-densify decision (all 4 entry points share one file)
    ("clear_capacity_scratch",             "densify/decide_actions_structure.slang", "clearCapacityScratch"),
    ("decide_actions_structure_pass1",     "densify/decide_actions_structure.slang", "decideActionsStructurePass1"),
    ("decide_actions_structure_threshold", "densify/decide_actions_structure.slang", "decideActionsStructureThreshold"),
    ("decide_actions_structure_pass2",     "densify/decide_actions_structure.slang", "decideActionsStructurePass2"),
    # common
    ("scan_block",           "common/scan.slang",                              "scanBlock"),
    ("scan_block_sums",      "common/scan.slang",                              "scanBlockSums"),
    ("add_block_offsets",    "common/scan.slang",                              "addBlockOffsets"),
    # mip-splatting (3D smoothing filter; 2D filter lives in project.slang)
    ("compute_3d_filter",    "mip/compute_3d_filter.slang",                    "computeMip3DFilter"),
)


# Subset of kernel attrs the viewer's ``_run_forward`` (see pipeline.py
# ``Pipeline._run_forward`` + ``_radix_sort_pairs`` + ``_scan``) actually
# dispatches. Skipping the rest saves a chunk of shader compile time and
# the associated program memory on ViewerPipeline construction — the viewer
# never runs backward, loss, densify, or Adam kernels.
VIEWER_KERNEL_ATTRS: frozenset[str] = frozenset({
    "clear_textures",
    "clear_tile_ranges",
    "identify_tile_ranges",
    "bin_count",
    "bin_emit",
    "bucket_count",
    "forward",
    "radix_init_global_hist",
    "radix_prepare_dispatch",
    "radix_upsweep",
    "radix_scan",
    "radix_downsweep",
    "scan_block",
    "scan_block_sums",
    "add_block_offsets",
})


def load_kernels(device: spy.Device, target: object,
                 viewer_only: bool = False) -> dict[int, str]:
    """Compile every kernel in :data:`KERNEL_SPECS` and set ``target.k_<attr>``.

    When ``viewer_only=True`` only the subset the render path uses (see
    :data:`VIEWER_KERNEL_ATTRS`) is compiled. Kernels outside that subset
    are NOT set on ``target`` — reading them raises AttributeError, which
    is the desired behavior (the viewer must not accidentally dispatch a
    training-only kernel against its shrunken buffer set).

    Returns a ``{id(kernel): entry}`` label map for debug markers (the
    dispatch path looks the marker name up by kernel identity).
    """
    labels: dict[int, str] = {}
    for attr, rel_path, entry in KERNEL_SPECS:
        if viewer_only and attr not in VIEWER_KERNEL_ATTRS:
            continue
        program = device.load_program(str(SHADERS_DIR / rel_path), [entry])
        kernel = device.create_compute_kernel(program)
        setattr(target, f"k_{attr}", kernel)
        labels[id(kernel)] = entry
    return labels
