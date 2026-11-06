# v3.2 full service-choice core grid

The v3.1 validation showed that the earliest-only service rule materially
changes several exact endpoint objectives and one focal NYNJ-use conclusion.
The registered core grid must therefore be rebuilt before any regret analysis
or final manuscript update.

This experiment uses the complete structural path set.  On every scheduled
leg, a shipment itinerary may use the earliest feasible departure or either
of the next two feasible departure events.  No service or capacity is added.
The model retains the same service-departure, arc-day, node-day, split-count,
and 10% minimum active-itinerary constraints.

The grid contains the existing 45 case-condition combinations: the registered
delay grid, three gateway-node capacity levels, and the compound disruption
for each destination.  Both gateway-access designs and both lexicographic
endpoints are solved, followed by exact NYNJ share bounds in every endpoint-
equivalent objective region.  Compatible proven v3.1 endpoints are reused.

Run on the remote server:

```bash
cd ~/check/china-us-multimodal-transport-paper
mkdir -p logs
conda activate chinaus311

nohup nice -n 10 bash scripts/run_full_service_choice_core_exact.sh \
  > logs/v32_full_service_choice_core_exact.log 2>&1 &

echo $! > logs/v32_full_service_choice_core_exact.pid
```

Monitor with:

```bash
process_id="$(cat logs/v32_full_service_choice_core_exact.pid)"
ps -p "$process_id" --ppid "$process_id" \
  -o pid,ppid,etime,%cpu,%mem,rss,cmd
tail -n 40 logs/v32_full_service_choice_core_exact.log
```

The run is checkpointed.  Restarting the same command skips proven endpoints
and proven, evaluator-consistent share bounds.
