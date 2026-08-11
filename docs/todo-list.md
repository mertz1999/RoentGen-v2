# Agentic Gen X-Ray — Complete Project TODO

This checklist translates `proposal2.docx`, the client messages, and the current
workspace state into an executable research plan.

## 1. Required project flow

The approved proposal requires this order:

```text
IU X-Ray data
    -> medical text generation
    -> synthetic radiology report / conditioning text
    -> RoentGen-v2 image generation
    -> image, text, and image-text alignment evaluation
    -> validated synthetic image-text pairs
```

Important decisions:

- [ ] Generate the synthetic medical text **before** generating its image.
- [ ] Pass the generated text directly through RoentGen-v2's own tokenizer and
      text encoder.
- [ ] Use MedCLIP after image generation to measure alignment; do not feed a
      MedCLIP embedding into RoentGen without a separately trained adapter.
- [ ] Do not use BioBERT as the text generator. BioBERT is an encoder, not a
      report-generation model.
- [ ] Use BioGPT, LLaMA-Med, or another explicitly documented medical generative
      LLM for the proposal's text-generation stage.
- [ ] Do not make MAIRA-2 part of the required pipeline. Generating another
      report at the end is not required by the proposal. MAIRA-2 may be added
      later as an optional round-trip validation experiment.
- [ ] Treat the existing Claude-generated captions as preliminary artifacts.
      Either regenerate them with the selected proposal-compliant model, or
      explicitly revise the methodology to say that an LLM was used to create
      standardized synthetic conditioning reports.
- [ ] Never report an experimental result unless its inputs, command/config,
      output file, model version, and seed are preserved.

## 2. Definition of done

The project is complete only when all of the following exist:

- [ ] A study-level, leakage-free train/validation/test split.
- [ ] A frontal-only dataset manifest with valid image-report mappings.
- [ ] A reproducible text-generation pipeline and its generated reports.
- [ ] A reproducible RoentGen-v2 baseline and LoRA generation pipeline.
- [ ] A fixed evaluation set of real and synthetic images.
- [ ] FID and KID results with identical preprocessing for both image sets.
- [ ] Text-quality results, including BERTScore and at least one clinical metric.
- [ ] MedCLIP matched-pair versus controlled-negative alignment results.
- [ ] A corrected, normalized HybridScore with justified weights.
- [ ] Baseline comparison and ablation study.
- [ ] A downstream experiment on a fixed, real-only test set.
- [ ] Confidence intervals and significance tests for the principal comparisons.
- [ ] Successful examples, failure examples, and a documented limitations section.
- [ ] Complete code, configs, dependency lock, manifests, raw metric outputs,
      figures, and result tables committed to Git.

## 3. Current verified state

- [x] IU X-Ray report CSV is available: 3,955 report rows.
- [x] IU X-Ray PNG collection is available: 7,470 images.
- [x] Existing Claude pipeline produced 3,955 non-empty conditioning captions.
- [x] RoentGen-v2 LoRA training code and recommended configuration exist in the
      adjacent `RoentGen-v2-mertz` repository.
- [x] A local metrics file contains 2,000 logged training steps and ends at a
      validation loss of approximately `0.09939`.
- [ ] Recover the raw metrics that support the separate step-6,000 plot.
- [ ] Recover the final LoRA weights or `checkpoint-6000`.
- [ ] Recover the claimed 546 generated images.
- [ ] Recover the command, real-image subset, and raw output supporting FID ~56.
- [ ] Recover any existing generated-report or evaluation CSVs.
- [ ] Commit the actual project code. The current Git history tracks only
      `.gitignore`; `generate_prompts.py` is untracked and `data/*` is ignored.

## 4. Phase 0 — Recover and inventory all existing artifacts

- [ ] Ask the client for a shared folder containing:
  - [ ] Final LoRA weights or `checkpoint-6000`.
  - [ ] The 546 synthetic images.
  - [ ] The exact 546 real images used for FID.
  - [ ] The inference prompt CSV.
  - [ ] All Colab/Kaggle notebooks.
  - [ ] Raw training logs and the final `metrics.json`.
  - [ ] FID command, package version, preprocessing, and raw output.
  - [ ] Any BioGPT/LLaMA/MAIRA outputs already produced.
- [ ] Generate SHA-256 checksums for every received archive and model artifact.
- [ ] Create `artifacts/inventory.csv` with file name, source, checksum, size,
      creation date, and purpose.
- [ ] Record which results are verified, claimed-but-unverified, or missing.
- [ ] Do not fabricate or copy example numbers from the ChatGPT conversation into
      the thesis result tables.

Deliverable: `artifacts/inventory.csv` and a verified status report.

## 5. Phase 1 — Freeze the experimental protocol

- [ ] Define the primary research claims:
  - [ ] The generated reports have acceptable linguistic and clinical fidelity.
  - [ ] RoentGen-v2 produces images of acceptable distributional quality.
  - [ ] Correct synthetic image-text pairs align better than negative pairs.
  - [ ] Hybrid selection identifies more useful pairs than no selection.
  - [ ] Selected synthetic data improves a downstream model on real test data.
- [ ] Define one immutable random seed list.
- [ ] Split at study/patient level before training or generation.
- [ ] Reserve the real test set and never use it for LoRA training, weight
      selection, threshold selection, or HybridScore tuning.
- [ ] Store the split in `data/manifests/splits.csv`.
- [ ] Choose the exact number of samples used in each experiment.
- [ ] Predefine primary and secondary metrics and the direction of improvement.
- [ ] Write the statistical analysis plan before viewing the final results.

Deliverable: `docs/experimental-protocol.md`.

## 6. Phase 2 — Prepare IU X-Ray correctly

- [ ] Start from the 3,955-report/7,470-image manifest.
- [ ] Exclude or resolve the 104 report rows with zero associated images.
- [ ] Obtain projection metadata and identify PA/AP frontal views reliably.
- [ ] Do not infer frontal versus lateral view only from the filename.
- [ ] Use frontal images for the core single-image experiment.
- [ ] Keep lateral/multiview studies in a separate optional experiment.
- [ ] Verify that every selected image has exactly one source report and split.
- [ ] Preserve the original image and create a deterministic 512x512 processed
      version using one documented transform.
- [ ] Use the same preprocessing for real and synthetic images during image
      metric calculation.
- [ ] Do not apply augmentations to validation or test images.
- [ ] Save `source_report`, `findings`, `impression`, `mesh_labels`, projection,
      image path, report ID, study ID, and split in the manifest.
- [ ] Produce summary tables for normal/abnormal and major finding frequencies.

Deliverables:

- `data/manifests/iu_xray_manifest.csv`
- `data/manifests/splits.csv`
- `results/data_summary.json`

## 7. Phase 3 — Generate the synthetic text first

### 7.1 Select and document the text model

- [ ] Select one primary model: BioGPT, LLaMA-Med, or another medical generative
      LLM that can be run reproducibly.
- [ ] Record the exact model ID, revision/commit, tokenizer revision, license,
      prompt template, decoding settings, and hardware.
- [ ] Explain the input to the text model. A defensible option is:

```text
source findings + impression + structured labels
    -> medical text-generation model
    -> new standardized standalone frontal-radiograph report
```

- [ ] Do not claim that a text-only model generated a report by looking at the
      image. It only sees the textual/structured fields supplied to it.

### 7.2 Generation constraints

- [ ] Generate present-tense, standalone radiology text.
- [ ] Keep only findings visible on a single frontal chest radiograph.
- [ ] Remove patient identity, age/sex unless explicitly part of the experiment,
      indication, recommendations, and non-visible clinical interpretation.
- [ ] Remove comparison language such as `stable`, `unchanged`, `previous`, and
      `interval` unless a prior-image experiment is explicitly implemented.
- [ ] Remove unresolved `XXXX` tokens.
- [ ] Preserve negation correctly: for example, do not change "no effusion" into
      "effusion."
- [ ] Limit the report using the actual RoentGen tokenizer, not `tiktoken`.
- [ ] Ensure the conditioning sequence fits the model's 77-token limit without
      truncating clinically important findings.
- [ ] Store the random seed and generation parameters for every report.

### 7.3 Validate the text output

- [ ] Repair the known preliminary-caption issues:
  - [ ] 13 captions estimated above 77 tokens.
  - [ ] 113 captions containing `stable`.
  - [ ] 10 captions containing `unchanged`.
  - [ ] Six captions longer than 50 words.
- [ ] Run automatic checks for temporal wording, recommendations, unsupported
      anatomy, empty output, duplicated output, and excessive length.
- [ ] Manually review at least 50-100 generated reports, sampled across findings.
- [ ] Record hallucination, omission, negation, and laterality errors.
- [ ] Compare generated text against its source report using:
  - [ ] BERTScore.
  - [ ] ROUGE-L and BLEU as secondary lexical metrics.
  - [ ] RadGraph F1, RaTEScore, CheXbert, or another clinical-fidelity metric.
- [ ] Select the final text set using only validation criteria.

### 7.4 Create RoentGen input

- [ ] Create `data/manifests/roentgen_inference_prompts.csv` with at least:

```csv
synthetic_id,source_report_id,prompt_variation,labels,split,text_model,text_seed
```

- [ ] Select an evaluation set with a documented balance of normal and major
      abnormal findings; do not take an uncontrolled random sample dominated by
      normal studies.

Deliverables:

- `src/text_generation/`
- `configs/text_generation.yaml`
- `data/manifests/generated_reports.csv`
- `data/manifests/roentgen_inference_prompts.csv`
- `results/text_quality.json`

## 8. Phase 4 — Train and validate RoentGen-v2 LoRA

- [ ] Pin the exact `stanfordmimi/RoentGen-v2` revision.
- [ ] Use only the training split for LoRA training.
- [ ] Use the fixed validation split for validation loss and checkpoint choice.
- [ ] Preserve the complete training configuration:
  - [ ] Resolution and image transform.
  - [ ] LoRA rank, alpha, and dropout.
  - [ ] Learning rate and scheduler.
  - [ ] Batch size and gradient accumulation.
  - [ ] Maximum steps/epochs.
  - [ ] Seed and GPU type.
  - [ ] Package versions and commit hash.
- [ ] Recover or rerun the 6,000-step training process.
- [ ] Fix metrics logging so resuming training appends to existing history rather
      than replacing prior metric arrays.
- [ ] Save periodic checkpoints and final LoRA weights.
- [ ] Plot raw/EMA training loss, validation loss, and learning rate from the
      preserved raw metrics.
- [ ] Choose the checkpoint using validation evidence, not the last training loss.
- [ ] Generate a small validation grid before the full inference run.
- [ ] Confirm that the pipeline can reload the saved LoRA weights from a fresh
      process and reproduce an image using a fixed seed.

Deliverables:

- `configs/roentgen_lora.yaml`
- `artifacts/checkpoints/` or an external artifact manifest
- `results/training_metrics.json`
- `results/figures/training_curve.png`

## 9. Phase 5 — Generate the synthetic images

- [ ] Compare at least these image generators with the same prompt set:
  - [ ] Base RoentGen-v2 without the project LoRA.
  - [ ] Proposed RoentGen-v2 with the selected LoRA.
- [ ] Use identical seeds, resolution, guidance scale, sampler, and inference
      steps for paired baseline comparisons.
- [ ] Generate one image per prompt for the primary experiment.
- [ ] Record every output in `generated_pairs.csv`:

```csv
synthetic_id,source_report_id,conditioning_text,labels,model,checkpoint,
seed,guidance_scale,inference_steps,sampler,image_path
```

- [ ] Verify image count, dimensions, readability, duplicates, corrupt files,
      and unexpected borders/artifacts.
- [ ] Preserve unsuccessful generations rather than silently regenerating only
      failures without documenting the selection process.
- [ ] If regeneration is permitted, record every attempt and the acceptance rule.

Deliverables:

- `data/synthetic/base/`
- `data/synthetic/proposed/`
- `data/manifests/generated_pairs.csv`

## 10. Phase 6 — Evaluate image quality

- [ ] Select a fixed real comparison subset using the real test/validation
      protocol and document exactly how it was sampled.
- [ ] Use equal sample counts when comparing generators.
- [ ] Apply identical preprocessing to real and generated images.
- [ ] Calculate FID and store the command, library version, features, sample
      count, preprocessing, and raw output.
- [ ] Calculate KID with repeated subsets and report mean and standard deviation.
- [ ] Consider a radiology-domain feature encoder as a secondary analysis because
      standard Inception features were trained on natural images.
- [ ] Repeat or bootstrap the comparison to quantify uncertainty.
- [ ] Compare base RoentGen-v2 and proposed LoRA under the same protocol.
- [ ] Do not interpret FID ~56 as good or bad without a same-protocol baseline.

Deliverables:

- `src/evaluation/image_quality.py`
- `results/image_quality_raw.json`
- `results/tables/image_quality.csv`

## 11. Phase 7 — Evaluate image-text alignment

- [ ] Pin the MedCLIP model and preprocessing revision.
- [ ] Compute normalized image and text embeddings.
- [ ] Calculate cosine similarity for every correct synthetic pair.
- [ ] Create controlled negative pairs:
  - [ ] Random mismatched reports.
  - [ ] Hard negatives with similar normal/abnormal status but different findings.
  - [ ] Multiple independently seeded shuffles.
- [ ] Avoid treating semantically equivalent reports from different patients as
      definitive negatives.
- [ ] Report matched and negative means, standard deviations, medians, and 95%
      bootstrap confidence intervals.
- [ ] Report retrieval metrics or matched-versus-negative AUROC in addition to raw
      cosine similarity.
- [ ] Run an appropriate paired permutation or Wilcoxon test.
- [ ] Plot the matched and negative score distributions.

Deliverables:

- `src/evaluation/alignment.py`
- `results/alignment_per_pair.csv`
- `results/alignment_summary.json`
- `results/figures/alignment_distribution.png`

## 12. Phase 8 — Define the HybridScore correctly

The client's example was:

```text
alpha * MedCLIP + beta * (1 - FID) + gamma * BERTScore
```

Do not implement that expression literally. FID is an unbounded, dataset-level
metric, so `1 - FID` is neither normalized nor a per-pair score.

### 12.1 Method-level HybridScore

- [ ] Normalize each method-level component using baselines fixed on the
      validation set:

```text
A = normalized alignment score, higher is better
Q = normalized image quality derived from FID/KID, higher is better
T = normalized clinical text-quality score, higher is better

MethodHybrid = alpha*A + beta*Q + gamma*T
alpha + beta + gamma = 1
```

- [ ] Define the normalization formula before evaluating the test set.
- [ ] Do not use test-set extrema to normalize test results.
- [ ] Choose weights using validation data, expert ranking, or downstream utility.
- [ ] Run a sensitivity analysis over several reasonable weight combinations.
- [ ] Report all component metrics beside HybridScore so the combined number never
      hides a failure in one modality.

### 12.2 Pair-level selection score

- [ ] Use only per-pair metrics when filtering individual pairs:

```text
PairScore_i = lambda*normalized_MedCLIP_i
            + (1-lambda)*normalized_text_fidelity_i
```

- [ ] Do not include global FID in `PairScore_i`.
- [ ] Select the threshold on validation data.
- [ ] Report the acceptance rate and finding distribution before and after
      filtering.
- [ ] Ensure filtering does not remove most rare/abnormal cases.

Deliverables:

- `src/evaluation/hybrid_score.py`
- `configs/hybrid_score.yaml`
- `results/pair_scores.csv`
- `results/hybrid_sensitivity.csv`

## 13. Phase 9 — Baselines and ablation study

- [ ] Evaluate all methods on identical prompts, image counts, preprocessing, and
      metrics.
- [ ] Minimum baseline comparison:
  - [ ] Base RoentGen-v2 + generated text.
  - [ ] LoRA RoentGen-v2 + generated text.
  - [ ] Correct pairs versus shuffled pairs.
- [ ] Minimum ablation comparison:
  - [ ] Original/cleaned reference text versus generated synthetic text.
  - [ ] LoRA off versus LoRA on.
  - [ ] Hybrid filtering off versus on.
  - [ ] MedCLIP-only filtering versus combined pair filtering.
- [ ] Do not compare the numeric FID of this project directly with DALL-M.
      DALL-M augments clinical/tabular features and is conceptual background, not
      an equivalent image-text generator baseline.
- [ ] If R2Gen/M2Trans or another report-generation baseline is included, rerun it
      under the same data split rather than copying published numbers from a
      different protocol.

Deliverable: `results/tables/baseline_ablation.csv`.

## 14. Phase 10 — Downstream usefulness experiment

This is the strongest evidence that the generated pairs are useful.

- [ ] Select a reproducible chest-X-ray classification task and fixed label set.
- [ ] Use the same architecture, initialization policy, optimizer, preprocessing,
      training budget, and real test set for all experiments.
- [ ] Run at least:
  - [ ] A: real training data only.
  - [ ] B: real + all synthetic data.
  - [ ] C: real + HybridScore-selected synthetic data.
- [ ] Optionally add D: synthetic pretraining followed by real-data fine-tuning.
- [ ] Keep validation and test sets completely real.
- [ ] Use at least three random seeds if compute permits.
- [ ] Report AUROC, F1, precision, recall, sensitivity, and specificity as
      appropriate.
- [ ] Report per-finding results, not only a macro average.
- [ ] Calculate confidence intervals and test the main paired differences.
- [ ] Interpret all outcomes honestly, including cases where synthetic data hurts.

Deliverables:

- `src/downstream/`
- `configs/downstream/`
- `results/downstream_runs.csv`
- `results/tables/downstream_summary.csv`

## 15. Phase 11 — Qualitative and failure analysis

- [ ] Select examples by a predefined rule, not only by visual preference.
- [ ] Include high-, medium-, and low-scoring pairs.
- [ ] Include both normal and abnormal cases.
- [ ] For every example show:
  - [ ] Source findings/labels.
  - [ ] Generated conditioning report.
  - [ ] Generated X-ray.
  - [ ] MedCLIP and text-fidelity scores.
  - [ ] Human assessment and identified errors.
- [ ] Categorize failure cases: missing finding, added finding, wrong laterality,
      device error, anatomy artifact, negation error, text truncation, or normal
      collapse.
- [ ] If possible, obtain a blinded review from a radiologist or qualified medical
      reviewer and document the rubric.
- [ ] State clearly that generated images are research artifacts and not suitable
      for clinical diagnosis.

Deliverables:

- `results/qualitative_review.csv`
- `results/figures/qualitative_examples/`
- `docs/failure-analysis.md`

## 16. Phase 12 — Repair proposal citations and terminology

- [ ] Correct the DALL-M reference to Hsieh et al., arXiv `2407.08227` / the 2025
      journal article. The currently cited arXiv `2305.01829` is unrelated.
- [ ] Correct MedCLIP to Wang, Wu, Agarwal, and Sun, arXiv `2210.10163`.
- [ ] Correct BioGPT to Luo et al., *Briefings in Bioinformatics*, 2022,
      arXiv `2210.10341`.
- [ ] Correct Jing et al.'s report-generation paper details to ACL 2018.
- [ ] Verify every remaining title, author list, venue, year, DOI, and arXiv ID
      against the primary publication.
- [ ] Remove duplicated English keywords in the proposal.
- [ ] Use one consistent term for each artifact:
  - `source report`
  - `generated synthetic report` or `conditioning report`
  - `synthetic X-ray`
  - `validated synthetic image-text pair`
- [ ] Do not call a cleaned/paraphrased source report an image-generated report.

Useful primary references:

- BioGPT: <https://arxiv.org/abs/2210.10341>
- MedCLIP: <https://arxiv.org/abs/2210.10163>
- MAIRA-2, optional only: <https://arxiv.org/abs/2406.04449>
- RoentGen-v2: <https://arxiv.org/abs/2508.16783>
- KID: <https://arxiv.org/abs/1801.01401>
- DALL-M: <https://arxiv.org/abs/2407.08227>

## 17. Phase 13 — Repository and reproducibility structure

- [ ] Organize the repository as follows:

```text
agentic-gen-xray/
  README.md
  todo-list.md
  pyproject.toml or requirements.txt
  configs/
  src/
    data/
    text_generation/
    image_generation/
    evaluation/
    downstream/
  data/
    manifests/
  artifacts/
  results/
    figures/
    tables/
  docs/
  tests/
```

- [ ] Keep raw images, checkpoints, and large generated assets out of Git.
- [ ] Commit small manifests, configs, scripts, checksums, and metric outputs.
- [ ] Replace hard-coded absolute paths with configuration values or CLI options.
- [ ] Add environment setup and one-command instructions to `README.md`.
- [ ] Add validation commands for manifests, model loading, metric schemas, and
      deterministic sample generation.
- [ ] Record Git commit hashes in every experiment result.
- [ ] Create a final release tag only after reproducing the tables from a clean
      environment.

## 18. Phase 14 — Thesis result package

- [ ] Produce the following final tables:
  - [ ] Dataset and split statistics.
  - [ ] Text-generation quality.
  - [ ] Image FID/KID comparison.
  - [ ] MedCLIP matched-versus-negative alignment.
  - [ ] HybridScore components and sensitivity.
  - [ ] Baseline and ablation results.
  - [ ] Downstream performance.
- [ ] Produce the following final figures:
  - [ ] Training/validation curves.
  - [ ] Alignment score distributions.
  - [ ] HybridScore threshold/acceptance analysis.
  - [ ] Successful and failed qualitative pairs.
  - [ ] Downstream comparison with confidence intervals.
- [ ] Write the results without assuming the proposed method must win.
- [ ] Separate observations, statistical evidence, interpretation, and limitations.
- [ ] Include exact model/config/data versions in the implementation section.
- [ ] Include ethics, intended research use, dataset limitations, and non-clinical
      status.

## 19. Optional MAIRA-2 round-trip validation

This is optional and should not block proposal completion.

- [ ] Feed each generated X-ray to MAIRA-2.
- [ ] Generate a reconstructed findings report.
- [ ] Compare it with the original generated conditioning report using BERTScore
      and a clinical metric such as RadGraph F1/RaTEScore.
- [ ] Treat the result as an additional round-trip consistency experiment, not as
      the proposal's required final report-generation stage.
- [ ] Document MAIRA-2's inputs, optional lateral/prior inputs, prompt, revision,
      decoding settings, and limitations.

## 20. Immediate six-hour client-update plan

The complete thesis experiment cannot honestly be finished in six hours, but a
credible progress package can be produced.

- [ ] Hour 0-1: request/recover checkpoint, 546 real/synthetic images, notebooks,
      prompt mapping, and raw FID evidence.
- [ ] Hour 1-2: build the artifact inventory, freeze terminology, and define the
      corrected experiment protocol.
- [ ] Hour 2-3: validate the existing 3,955 captions and create a clean RoentGen
      inference manifest.
- [ ] Hour 3-4: reproduce a small fixed-seed image-generation sample if the LoRA
      weights are available.
- [ ] Hour 4-5: run the metrics that are possible with the recovered artifacts;
      otherwise prepare their executable configs and input manifests.
- [ ] Hour 5-6: deliver a verified status table, corrected HybridScore definition,
      sample outputs, blockers, and the next GPU execution plan.

Do not describe the project as complete in this update. Report clearly which
items are verified, running, missing, or blocked by unavailable artifacts/GPU.

## 21. Final sign-off checklist

- [ ] A fresh environment can reproduce one synthetic text and its image.
- [ ] A fresh environment can reproduce every final metric table from manifests.
- [ ] No train/test leakage is present.
- [ ] All claims are supported by raw results.
- [ ] HybridScore normalization and weights are fully defined.
- [ ] Required proposal stages are complete in the correct order.
- [ ] Optional MAIRA-2 work is clearly labeled optional.
- [ ] All bibliography entries resolve to the intended primary publications.
- [ ] All final tables and figures trace back to scripts and result files.
- [ ] The limitations section discusses sample size, IU X-Ray bias, metric
      limitations, clinical fidelity, and lack of clinical deployment validation.
