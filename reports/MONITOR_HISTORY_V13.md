# Durable monitor history — 2026-10-09

Changes are limited to the CPU observer, tests, and release documentation. No changes to repro/, configuration YAML, training math, optimizer, batch plan, dataset or checkpoint identity. training_hash and recipe_hash match the published V12 checkout in .updates/publisher.

Laptop viewer lives outside this training bundle in D:/Chay_Paper/SignLanguage-Monitor. It migrates the existing metric cache, persists dev summaries/checkpoints/speed, joins sanitized log tails into a SQLite journal, marks collection gaps, offers paginated train/dev/epoch filters and TXT download, restores metrics while offline, and backfills W&B scalar history. Distinct experiments and observer runs are kept separate. Log files and caches contain no API key. A saved Windows credential was used in memory for read-only validation; it was not printed, changed, or written to an artifact.

Validation:
- 26 viewer/credential/persistence tests passed. Covers shutdown/reopen, stale tail vs reset, metric updates without train advancement, offline reopening, exact teacher campaign path, separate campaign epoch metrics, log overlap and gaps, archive recovery, HTTP filters/TXT download and redirect authentication removal.
- 43 observer/background/supervision tests passed, including 7 console archive tests. Complete source bytes and UTF-8/CRLF boundaries tested; partial lines held; retry retains cursor; archive path guards; one older host console backfilled per poll; active source preserved.
- Live W&B read-only scans returned completed epoch summaries 1–7 for er49ebw0 and the named-file query read live_log_tail.txt. Initial parsing only recognized the seed0 output path; real API validation found the suite/campaign path, which is now supported and regression tested.
- Browser verified train/dev table, persistent history after restart, dev filtering and older-log navigation. At the narrow 319px viewport, page scrollWidth was 313px; tables scroll internally. Desktop reviewed at 1440x1000. Screenshot outside the training bundle shows actual W&B epoch records, not synthetic scores.

Deployment limit:
The teacher worker stays pinned while running. Publishing V13 does not hot-swap the currently running observer/trainer. The existing supervisor adopts it at its normal boundary after the worker ends or fails. Only then does the console archive start uploading, including older console files still on the host. Until then the viewer can recover W&B epoch scalars and locally collected log tails, but cannot recover overwritten raw tails from W&B. Per-video dev progress is not currently logged by the trainer. No claim that the teacher already runs V13 or that all old raw logs are available now.
