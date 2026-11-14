# CROWD observed-frame block audit

This tool audits the complete accessible source-video inventory, independently of
CTC target eligibility. It does not train a model or correct source labels.
Technical verification is distinct from semantic verification.

## Processing contract

- Enumerate every MP4 member of the original ZIP and match original morpheme JSON
  by canonical video ID. Duplicate source names receive a deterministic suffix.
- Decode each video to obtain actual presentation times. Select unique observed
  source frames against a 15 Hz time grid and include the actual final frame.
  The plan retains every decoded source PTS, while coordinate extraction uses
  only the selected actual frames. VFR bursts may be thinned even when average
  FPS is below 15. Uniform low-rate inputs retain all source frames. No
  interpolation or duplicate frames are represented as observations.
- Partition selected frames chronologically into disjoint groups of at most 15.
  The final partial group is retained. A block is a recovery unit, not a letter.
- Reuse old verified MediaPipe features only when archive member, CRC, SHA,
  extractor versions and actual selected PTS match. Old cropped features alone
  do not establish whole-video coverage.
- New extraction uses the existing `landmarks.process_frame` contract without
  swapping hands or mirroring. NPZ retains the compatible 53-point schema;
  geometry analysis uses only the two 21-point hands. Contact sheets crop the
  observed hands from actual decoded images.
- Record source frame indices, selected indices, actual PTS, validity masks,
  feature provenance, tracker context IDs, schema and SHA. Reopen saved outputs
  before assigning `VERIFIED_TECHNICAL`.
- On resume, verify completed artifacts and reuse them. Decode preceding context
  when needed for tracking without counting that context as new observations.
- A recoverable block error has at most 11 total extraction attempts. Attempt
  counts are persisted before processing and survive restarts. Source/probe
  stages have separate durable budgets. Journal I/O and artifact readback retry
  independently without repeating successful landmark extraction.
- Structural errors stop the affected unit immediately; a failed unit does not
  stop subsequent videos. Temporary cleanup failures retain the temporary file
  and do not invalidate already verified technical outputs.

Wrist movement is measured in palm lengths per second with actual time deltas.
Finger angles and shape-change rates describe observed geometry. The thresholds
for stable/transition spans are heuristics recorded in `schema.json`. Whole-video
spans may cross block boundaries; tracking-context discontinuities are marked.
No validated handshape-to-jamo rule is available, so candidate letters remain
empty and semantic status remains `UNCERTAIN`. Original labels and decomposed
jamo are review context, not independent evidence that the gestures match.

## Outputs

Store outputs outside the repository on D:, for example
`crowd/outputs/crowd-block15-audit-v1`:

- `inventory.json`, `schema.json`: immutable source inventory and compatibility
  contract.
- `videos/<ID>/plan.json`: complete source PTS and selected source indices.
- `videos/<ID>/state.json`: durable per-stage/per-block status, attempts, errors,
  recovery, paths and hashes.
- Per-block NPZ and hand contact JPEG; `summary.json` and `review.html` combine
  original labels, jamo and whole-video geometry observations.
- `audit.json`, `failure-and-unprocessed-lists.json`: disjoint coverage counts
  and explicit failures/unprocessed IDs. Unknown block totals for videos without
  a decoded plan must not be presented as complete coverage.
- `source-guard.json`, `source-integrity.json`: original ZIP SHA checks.

Only this run's individual, CRC-verified temporary MP4 may be removed, after
all result blocks verify and its decoder is closed. Existing temporary MP4s,
original ZIPs, old features/checkpoints and unsuccessful partial files remain.

## Reproduction

Run from this branch's worktree using the existing Conda interpreter:

```powershell
$env:TEMP = 'D:\UNIVERSITY\UAI_project2\FILE2\sign-language-recognition\.tmp-issue2'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\UNIVERSITY\DOWNLOAD\MINI_CONDA\REAL_MINICONDA\envs\sign-language\python.exe' `
  tools/process_crowd_blocks.py `
  --data 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd' `
  --previous 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\mediapipe-ctc-performance\data-expansion-01' `
  --output 'D:\UNIVERSITY\UAI_project2\sign-language-data\fingerspelling\crowd\outputs\crowd-block15-audit-v1' `
  --workers 3
```

Repeat the same command to resume. `--ids ID ...` is a pilot filter only; the
inventory and final audit still describe all source videos. `--audit-only`
reopens artifacts without extraction. A lock prevents concurrent coordinators
from writing to the same output directory. No package changes are required.

Synthetic recovery tests and actual video processing are separate evidence.
Final measured counts and regression results are recorded after the full run.

## Actual pilot and recovery evidence

Two source videos produced 18 verified blocks. `NIA_SL_FS0450_CROWD19_F`,
previously excluded for low FPS, has 105 decoded source frames, metadata FPS
13.6659436 and observed mean rate 13.5652174. Its VFR selection contains 96 unique
observed frames in 7 blocks, including the final 6-frame block.
`NIA_SL_FS0070_CROWD18_F` has 211 source frames and 158 selected observations in
11 blocks. The original pilot reused 109 old feature observations across the
two videos. These counts distinguish source frames from selected observations.

The first pilot incorrectly classified the second video as failed after a
Windows `WinError 32`: its decoder still held the newly created disposable MP4
open during cleanup. All 11 block outputs had already verified. The fix closes
the decoder before cleanup and reports cleanup separately. The subsequent full
run re-read both videos' verified outputs and reported SUCCESS with
`resumed_without_extraction=true`; it did not regenerate their blocks.

Independent review found and verified fixes for unsupported numeric/English
label handling, joining old workers before requeueing, durable dispatch failure
budgets, source reuse without consuming extraction retries, and ZIP SHA checking
in audit-only mode. Unsupported jamo strings remain original review context;
they do not invalidate technically processed videos.
