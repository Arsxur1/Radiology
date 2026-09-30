# Resume point — D45 (sinus label requires organised P waves)

Status: implemented and consistency-checked (60/60 vs dev), NOT yet validated, NOT active on the branch.
The candidate code is in `d45_candidate.patch` (apply with `git apply docs/wip/d45_candidate.patch`);
the pre-registration (written before held-out data) is in `d45_prereg.txt`.

Held-out results so far (pre-registered criteria: fires on <= 1.5% of true sinus per set; >= 20% of
non-sinus rhythms called sinus, pooled):
- Fresh Georgia (1,384 unused records): true sinus 4/617 = 0.65% (pass); non-sinus 9/44 = 20%.
- New Chapman/Ningbo sample (16 never-used directories, 1,600 records): run was in progress, not scored.

Next steps:
1. Re-create the held-out Chapman sample (directory list chosen with seed 20261001 from directories not
   used before; see scratch scripts) and run the full pipeline with the patch applied.
2. Score both sets against the criteria; ship (commit patch + D45 in DECISIONS/VALIDATION) or document
   as negative.
Open product decision noted in DECISIONS D42/D43: AF operating point of the pre-QRS consistency feature.
