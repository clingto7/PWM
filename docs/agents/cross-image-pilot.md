# Cross-image pilot: operational lessons

The acceptance scope and split rules live in [Spec #14](https://github.com/clingto7/PWM/issues/14). This note records pitfalls found while running the first development-game images. It does not change the spec's thresholds or authorize image-specific rules.

## Judge the image, not the process exit

An observation stage can finish and save valid artifacts while selecting a road, foreground fragment, or incomplete subject. For every eligible development image, inspect the source screenshot, selected mask, candidate masks, and selection evidence before recording success. Record abstentions and incorrect or partial selections as failures. Keep those images in the denominator. A syntactically valid manifest and a `complete` stage status prove artifact production, not subject correctness.

When a selection fails, first ask whether SAM produced a usable full-subject proposal. If no candidate covers the subject, change proposal coverage and retain prompt/multimask provenance. If a good candidate exists but loses selection, investigate ranking. This separates generation failure from selection failure and avoids adding per-game or per-image thresholds to hide the symptom.

## Preserve evaluation independence

Keep source URL and image SHA for each screenshot. Review eligibility before scoring, and use manual 2D annotations only for scoring. Freeze algorithm, parameters, failure criteria, numeric thresholds, and aggregate target from development results before opening held-out images. Retain failed eligible scenes and their diagnostics; do not replace them after seeing outputs.

## Run on A800 reproducibly

Before a substantial batch, check GPU processes, shared capacity, assigned-directory size, and the pinned runtime environment. Probe imports needed by the full chain, not just local unit tests: the first-image run encountered a Torch/torchvision `torchvision::nms` mismatch despite passing portable tests. Record any process-only runtime workaround with the run; source fixes still belong on the Mac.

Publish a locally reviewed commit before updating the server checkout. For Git bundles, check size before transfer; prefer a bundle with the already-present integration revision as prerequisite when transferring one follow-up commit. Verify transferred bytes or SHA before `git fetch`, remove incomplete temporary transfers, and check the server checkout is clean before and after. Keep all runtime artifacts within the assigned workspace and never edit source on the server.
