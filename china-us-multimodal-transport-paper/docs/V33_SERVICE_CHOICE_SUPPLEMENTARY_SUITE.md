# v3.3 Selectable-service supplementary suite

This suite re-estimates every supplementary result currently used in the
manuscript after the exact model was expanded from the earliest feasible
departure only to the earliest feasible departure plus the next two published
departure events on every scheduled leg.

It uses the exhaustive structural path library and the proven v3.2 exact core
grid as its reference. The suite covers:

1. tardiness-penalty and holding-cost sensitivity;
2. baseline ocean-cost and Panama-transit-time sensitivity;
3. release-pattern robustness;
4. demand scaling;
5. local and regional inland-capacity tests;
6. epsilon-constrained intermediate solutions; and
7. the final matched zero-penalty and activation-criterion addenda.

All runs use exact lexicographic MILP endpoints and exact endpoint-equivalent
NYNJ share bounds. Existing v2.2-v2.8 outputs remain provenance records for the
earliest-only formulation and must not be combined with v3.2 or v3.3 results in
the revised manuscript.

Run from the project root with:

```bash
mkdir -p logs
nohup nice -n 10 bash scripts/run_service_choice_supplementary_suite.sh \
  > logs/v33_service_choice_supplementary.log 2>&1 &
echo $! > logs/v33_service_choice_supplementary.pid
```

Resume by running the same command. Each optimization module uses checkpoint
files and skips already proven tasks.
