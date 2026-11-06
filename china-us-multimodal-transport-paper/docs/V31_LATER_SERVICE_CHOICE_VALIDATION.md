# v3.1 later-service choice validation

## Purpose

This is a targeted validation of the earliest-reachable-service assumption.
It does not add services or capacity.  Under the expanded rule, every
scheduled leg may choose its earliest feasible departure or either of the next
two feasible departure events.  The choices enter the exact MILP as distinct
shipment itineraries and remain subject to the same service-departure,
arc-day, node-day, splitting, and minimum-share constraints.

## Registered conditions

- Chicago: 288 h West Coast delay;
- Memphis: 288 h West Coast delay;
- Columbus: 216 h and 240 h West Coast delay;
- Chicago, Memphis, and Columbus: 35% remaining West Coast gateway-node
  capacity without added delay; and
- Columbus: 30% remaining regional West Coast inland capacity plus 168 h
  delay.

For every condition, both gateway-access designs (with and without NYNJ) and
both exact lexicographic endpoints are solved.  The 32 historical
earliest-only endpoints are reused from proven, evaluator-consistent records.
The expanded rule therefore adds 32 endpoint tasks, with two HiGHS calls per
endpoint when both lexicographic stages finish.

The active-itinerary limit remains three per shipment.  A structural route
paired with a different sequence of scheduled departures is a distinct
itinerary for that limit and for the 10% minimum active-itinerary share.

## Server command

```bash
cd ~/check/china-us-multimodal-transport-paper
mkdir -p logs
conda activate chinaus311

nohup nice -n 10 bash scripts/run_later_service_choice_validation.sh \
  > logs/v31_later_service_choice_validation.log 2>&1 &

echo $! > logs/v31_later_service_choice_validation.pid
```

Monitor with:

```bash
process_id="$(cat logs/v31_later_service_choice_validation.pid)"
ps -p "$process_id" -o pid,etime,%cpu,%mem,cmd
tail -n 40 logs/v31_later_service_choice_validation.log
```

The runner is checkpointed and safe to restart with the same command.  It
writes objective values, NYNJ shares, gaps, runtimes, full shipment
allocations, selected departure ranks, realised service legs, current-versus-
expanded endpoint changes, and gateway values to
`results/v31_later_service_choice_validation`.

## Decision rule

- If primary objective values, endpoint NYNJ use, and with-versus-without-NYNJ
  values remain practically unchanged, the paper may report that the tested
  two-departure relaxation did not alter the focal conclusions.
- If a focal endpoint changes materially, analyses that depend on that
  endpoint must be rebuilt under the expanded service-choice model before the
  regret experiment is designed.
- The test does not justify a claim that every later departure has been
  enumerated.
