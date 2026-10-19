# Open-source release: Legal review handoff

**Internal review material, 2026-10-15.** This is a technical inventory, not
release approval. Keep internal GitLab links and process notes out of a public
export unless their inclusion is approved. The public-facing inventory is
[NOTICE.md](../NOTICE.md).

## Requested release changes

- The root [LICENSE](../LICENSE) contains the EA license supplied for this
  project, with the 2026 copyright year.
- [README.md](../README.md) contains the supplied contribution notice and CLA
  link, an explicit table of all three research datasets, and a pointer to the
  third-party notice instead of the broad acknowledgement wording.
- EA-authored source headers use the supplied text:
  `Copyright (c) 2026 Electronic Arts Inc. All rights reserved.` Source history
  begins in May 2026 and all recorded source-commit years are 2026. Existing
  nonignored untracked sources are included in the header pass. Original
  authoring years and ownership still need author confirmation where Git does
  not establish provenance.
- The shader contract generator emits the same notice, so regenerating its
  four Slang outputs does not drop it.
- In the three known mixed-origin files below, the EA notice is explicitly
  limited to EA-authored wrappers, ports, or modifications. Upstream notices
  and the copied code are retained; these are not claimed as wholly EA-owned.

## Specific third-party-code locations for review

The two confirmed GitLab permalinks below use the locally recorded GitLab
revision `0a4263ec5a11b5c480319adefb97f4cda628d0f2`. At review time their source
blobs matched the corresponding local files before header insertion. These
links intentionally show the pre-notice code, so their line references are
stable. This verifies the local remote-tracking evidence, not live server
availability or reviewer permissions.

| Material | Current local file | GitLab review location |
| --- | --- | --- |
| Verbatim INRIA metric functions, obtained through SAD-GS | [scripts/plots/image_metrics.py](../scripts/plots/image_metrics.py) | [Import provenance](https://gitlab.ea.com/seed/slang3dgstrainer/-/blob/0a4263ec5a11b5c480319adefb97f4cda628d0f2/scripts/plots/image_metrics.py#L3-12); [PSNR and SSIM functions](https://gitlab.ea.com/seed/slang3dgstrainer/-/blob/0a4263ec5a11b5c480319adefb97f4cda628d0f2/scripts/plots/image_metrics.py#L37-89) |
| Slang port of Thomas Smith's GPUSorting | [shaders/sort/device_radix_sort.slang](../shaders/sort/device_radix_sort.slang) | [Port attribution and scope](https://gitlab.ea.com/seed/slang3dgstrainer/-/blob/0a4263ec5a11b5c480319adefb97f4cda628d0f2/shaders/sort/device_radix_sort.slang#L1-30) |
| Adapted INRIA evaluator, retaining its copyright header | [src/trainer/frontend/evaluation/metrics.py](../src/trainer/frontend/evaluation/metrics.py) | **Pending push:** [local base-commit location](https://gitlab.ea.com/seed/slang3dgstrainer/-/blob/1a9045655b9efb0691d24cf71b55ece8a2e6eab6/src/trainer/frontend/evaluation/metrics.py#L1-105) |

The evaluator was added in local commit
`1a9045655b9efb0691d24cf71b55ece8a2e6eab6`; none of the locally known GitLab
remote branches contains that commit. Do not present its URL as available until
the approved branch is pushed and the link checked. The older
`src/trainer/common/metrics.py` on those remote branches is a different file,
not a substitute link to the copied evaluator. No commit or push is performed
by this preparation.

After pushing the review branch, also provide a permalink to the reviewed
notice and refresh links if the source layout changes.

## Non-EA-source modification details for the required Legal email

The copyright instructions require details of modifications to non-EA-owned
code to be sent to [Opensource-legal@ea.com](mailto:Opensource-legal@ea.com).
The information below is prepared for that email; **no email has been sent**.

| File | Upstream and retained material | Local modifications / additions |
| --- | --- | --- |
| [src/trainer/frontend/evaluation/metrics.py](../src/trainer/frontend/evaluation/metrics.py) | INRIA / GRAPHDECO 2023 evaluator; original header retained | Evaluation directories are collected and sorted by numeric iteration suffix before evaluation. A scoped EA-modifications notice and license pointer have been added; the evaluator's existing body is unchanged by this release-preparation work. |
| [scripts/plots/image_metrics.py](../scripts/plots/image_metrics.py) | PSNR and SSIM helpers explicitly marked verbatim from INRIA via SAD-GS | Local PNG loading, camera/iteration selection, holdout selection, aggregation, reporting, and CLI wrap the copied helpers. A scoped EA-wrapper notice and upstream attribution have been added; the copied functions are unchanged by this work. |
| [shaders/sort/device_radix_sort.slang](../shaders/sort/device_radix_sort.slang) | Thomas Smith GPUSorting reduce-then-scan WGE16 path, MIT | Slang port; 64-bit key represented by two 32-bit words; Gaussian-index payload; eight radix passes; trainer-specific buffer contracts, capacity handling, GPU-resident count, and indirect-dispatch integration. A scoped EA-port notice and original copyright/license pointer have been added; shader logic is unchanged by this work. |

The exact upstream code revisions and the SAD-GS intermediary revision were
not recorded in the existing source. License snapshots are identified in
[NOTICE.md](../NOTICE.md); they must not be mistaken for proof of the code's
original import revision. The modification descriptions are a starting point
for author confirmation, not a complete upstream diff or provenance warranty.

## Decisions still required before publication

1. **Copied INRIA code:** determine compatibility with the intended release,
   including research/non-commercial restrictions on the evaluator and PSNR
   source. Preserve the separate SSIM / Evan Su MIT provenance identified in
   Section 6 of the upstream license. A notice alone is not clearance; decide
   whether permission, replacement, or exclusion is needed.
2. **Dataset terms:** confirm archive-specific terms for Mip-NeRF 360 (including
   flowers/treehill), Tanks & Temples, and Deep Blending. The current Tanks &
   Temples page contains both CC BY 4.0 language and a restrictive research-only
   grant; do not resolve that conflict by assumption. Deep Blending terms have
   not been verified from an accessible authoritative copy.
3. **Optional external materials:** review the applicable STBN texture terms
   (the current upstream document has both non-commercial and commercial
   sections), pretrained model-weight terms, and any bundled licenses in
   COLMAP/GLOMAP or installed Python wheels. The notice is a direct-use
   inventory, not a resolved transitive SBOM.
4. **Tracked previews:** confirm redistribution rights for the dataset-derived
   bicycle renders and experimental plots under [assets](../assets).
5. **Git history versus clean snapshot:** no raw datasets are tracked in the
   current tree, but history reachable from HEAD includes 147 cactus scan paths
   and 23 synthetic Suzanne paths, deleted in commit `9d65371` on 2026-07-26.
   Confirm ownership and whether to publish that history or a separately
   prepared approved snapshot. No data contents were inspected and no history
   was rewritten.
6. **Other provenance:** confirm that algorithm citations and parity comments
   elsewhere do not conceal additional source reuse. For example,
   [projection_backward.slang](../shaders/rasterize/tile_binned/projection_backward.slang)
   says it mirrors a faster-gaussian-splatting preprocessing design. This alone
   does not establish a copied implementation, nor does absence of a copy
   notice prove independent authorship.

## Technical validation

- All 349 nonignored Python, Slang, shell, and PowerShell source files have the
  supplied copyright notice. Five shell shebangs remain first. Before the test
  run, the existing working-tree source contents were checked against an
  in-memory pre-edit baseline: only notice insertions and the generator's
  emitted banner changed, allowing line-ending normalization.
- All 262 Python sources parse. The four generated shader headers match their
  generator, and the supplied CLA paragraph matches the README exactly.
- The bundled INRIA license and Thomas Smith MIT notice match their pinned
  upstream snapshots. Temporary wheel and source-distribution builds include
  all four license/notice files with matching contents; this internal handoff
  document is not included in those package archives.
- Focused CPU validation: **64 passed, 1 deselected**. This covers the contract
  generator, source architecture, CPU metrics, and evaluation harness. The
  deselected test requires a Vulkan device.
- Full trainer and evaluation-harness suites: **775 passed, 21 failed, 230
  skipped**. The repository is **not test-clean**. Failure groups are listed
  below; no unrelated runtime or test-expectation changes were made to turn
  this legal-preparation task into a behavior-change patch.

| Full-suite failures | Count | Area |
| --- | ---: | --- |
| Frozen defaults and evaluation-enable expectations differ from the current settings | 6 | [CLI defaults](../tests/test_cli_defaults.py), [schedule configuration](../tests/test_config_types.py), [evaluation toggle](../tests/test_evaluation_toggle.py) |
| Synthetic COLMAP fixture lacks a sparse reconstruction | 4 | [COLMAP tests](../tests/test_colmap.py) |
| Vulkan device creation fails before shader reflection | 1 | [Contract reflection](../tests/test_contracts.py) |
| The metric-direction table does not recognize `lpips_alex` | 1 | [Paper statistics](../tests/test_paper_suite_stats.py) |
| Capture test passes an obsolete `it` argument instead of an iteration context | 1 | [Profiling boundary](../tests/test_profiling_boundaries.py) |
| Training test doubles target obsolete pipeline/loop state, including missing STBN RNG state | 8 | [Training application](../tests/test_training_application.py) |

The relevant settings, profiler, statistics, and COLMAP loader files match HEAD
apart from the copyright headers. The training-loop entry method also matches
its HEAD AST; pre-existing STBN changes were covered by the working-tree baseline
check. Those failures are not evidence of a runtime change by this notice pass,
but they remain unresolved repository issues and should be tracked separately.
The whitespace check also retains the pre-existing shader trailing whitespace
and a harmless blank line after the notice in a formerly empty initializer.

## Suggested review message

> The preparation now includes the supplied EA license, source copyright
> headers, contribution/CLA notice, and a third-party notice covering Mip-NeRF
> 360, Tanks & Temples, Deep Blending, dependencies, and included third-party
> code. The file inventory above gives the source locations and modification
> details, with one evaluator permalink still pending the review-branch push.
> Please review the copied INRIA components and the remaining dataset/asset
> questions before approving publication. The Legal notification email and
> release approval remain outstanding.