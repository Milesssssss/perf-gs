# Sparse pixel supervision in Ultrafast

Ultrafast renders the full image and uses a spatiotemporal blue-noise mask to select 25% of the pixels for gradient updates. Balanced and Quality use every pixel. All three presets evaluate held-out views at their full selected resolution. [Preset measurements](benchmarks.md#current-presets)

## Mask download and terms

From the checkout, download the scalar masks once, then start training.

```sh
python scripts/data/download_stbn.py
python -m trainer.frontend.app --scene datasets/360_v2/bicycle --downsample 4 --preset ultrafast
```

The hub also offers the download when a preset needs the masks. The helper stores the masks, attribution and license in the local `.stbn/` cache. NVIDIA licenses this asset for non-commercial research and evaluation. Redistribution requires its license and attribution. For a separately sourced scalar volume, pass `--stbn-source file --stbn-file MASK.npy` and follow the terms for that asset. [NVIDIA license](https://github.com/NVIDIA-RTX/STBN/blob/main/License.txt) · [Scalar Spatiotemporal Blue Noise Masks](https://arxiv.org/abs/2112.09629)

## How sampling works

The NVIDIA mask consists of 64 slices, each 128 × 128 pixels. Ultrafast selects the lowest quarter of values across each slice and tiles the result across the image. The count varies from one 8 × 8 raster tile to the next. Each camera advances to the next slice on its next training visit. [Mask selection](../src/trainer/backend/render/stbn_table.py) · [Camera sampling](../src/trainer/frontend/sampling.py)

The forward pass renders every pixel. SSIM computes local statistics over its full 11 × 11 window, so neighboring pixels still provide context. When the loss writes the gradient of the rendered image, it zeros unselected pixels and scales selected gradients by four at 25% density. The rasterizer’s backward pass then visits the selected pixels in each tile. [SSIM shader](../shaders/losses/ssim_loss.slang) · [Rasterizer backward pass](../shaders/rasterize/tile_binned/backward_persplat.slang)
