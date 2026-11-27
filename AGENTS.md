## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context domain layout. See `docs/agents/domain.md`.

### Cross-image pilot

When running or evaluating Spec #14 across game screenshots, see
`docs/agents/cross-image-pilot.md`.

## A800 server usage

Use the `a800` SSH alias and the assigned PWM directory
`/KAIROS_vepfs-2/KAIROS_vepfs/liwenhao/agent/666777`. Follow the shared
`asus-host-context` Skill and the machine-wide remote development boundary:
write and publish source code only on the local Mac; keep remote work to
approved revisions, experiments, and their expected outputs.

At the end of every PWM work turn, check and report the A800 GPU count and
memory use, active GPU jobs, shared filesystem capacity and free space, and
the assigned directory's size. If the server is unreachable, say the live
check failed and label any older figures as a past snapshot. Check the same
figures before and after substantial remote runs. Keep runtime outputs and
unavoidable caches within the assigned directory or an approved storage
location; avoid filling the shared filesystem, root, or `/tmp`. Idle GPUs are
not an allocation: check current jobs and scheduling constraints before use.
