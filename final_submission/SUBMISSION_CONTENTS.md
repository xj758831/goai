# Clean Submission Contents

This directory was extracted from `goai_follow_499_baseline_submission` as a
standalone, non-destructive clean copy. The source directory was not deleted or
overwritten.

## Included

- The v57 entry point and every local Python module reachable through its
  import graph.
- The synchronized ROS tail navigator, its local imports, and the two bridge
  helpers selected at runtime.
- The official S10 MuJoCo XML, complete mesh tree, and shipped ONNX policy.
- The route, tail behavior, point-cloud shadow, and footprint configurations
  referenced by the retained code.
- The four runtime checkpoint/trace files declared in
  `BASELINE_MANIFEST.json`.
- The locked 33/33 success summary and its evidence manifest.
- English and Chinese reproduction instructions and a read-only verifier.

Some retained modules have historical `train_` or `search_` names. They remain
because the current expert controller imports classes, constants, or helper
functions from them at module load time. They are runtime dependencies in the
current architecture, not extra training outputs.

## Excluded

- Git history and editor/cache directories.
- Historical MP4/GIF files and partial recordings.
- Point-cloud JSONL traces, ROS command traces, and temporary state snapshots.
- Old training runs, candidate checkpoints, campaign outputs, and unrelated
  experimental configurations.
- Python bytecode and `__pycache__` directories.

`SUBMISSION_MANIFEST.sha256` covers every retained file except itself. Run
`sha256sum -c SUBMISSION_MANIFEST.sha256` from this directory to detect missing
or changed files.
