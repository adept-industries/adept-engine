# Worker monitoring contract

This process-level endpoint makes worker progress observable independently from
the Engine API. The Engine API's `/health` and `/ready` routes still describe a
different container and do not prove that background jobs are processing.

## Private scrape endpoint

PR 3 must scrape this exact target from Grafana Alloy:

```text
GET http://engine-worker:8001/metrics
job_name: adept-engine-worker
scrape_interval: 60s
scrape_timeout: 15s
```

The worker serves Prometheus text format directly. It needs no Grafana URL,
username or token. The endpoint is plaintext and unauthenticated because it is
restricted to the private Docker Compose network. Never add a host `ports:`
mapping or a Caddy route for it. `EXPOSE 8001` in the image is metadata only and
does not publish the port.

The non-secret worker settings are:

| Environment variable | Default | Validation | Purpose |
|---|---:|---:|---|
| `ENGINE_METRICS_BIND_ADDRESS` | `0.0.0.0` | 1–255 characters | Listen on the container network |
| `ENGINE_METRICS_PORT` | `8001` | 1–65535 | Private scrape port |
| `ENGINE_QUEUE_METRICS_INTERVAL_SECONDS` | `60` | 5–3600 seconds | Cached queue-query interval |
| `ENGINE_WORKER_THREADS` | `2` | `1` or `2` | Processing-thread count |

The path is fixed at `/metrics`. Keep the production bind address, port and
Alloy target consistent. These settings belong to the worker's application
environment, normally Compose and `/opt/adept/.env.production`; they do not
belong in `/opt/adept/.env.monitoring`. The monitoring file contains only the
Grafana credentials consumed by Alloy.

## Metrics

All metric names use the `adept_engine_worker_` prefix. `thread_slot` is the
stable configured slot (`1` or `2`), never the restart-specific database owner.
`job_type` is limited to registered handler types; any unexpected value becomes
`UNKNOWN`.

| Name | Type | Labels | Unit and meaning |
|---|---|---|---|
| `configured_threads` | Gauge | none | Configured processing threads |
| `process_start_time_seconds` | Gauge | none | Process start Unix timestamp |
| `shutdown_requested` | Gauge | none | `1` after SIGTERM/SIGINT, otherwise `0` |
| `thread_alive` | Gauge | `thread_slot` | `1` while the consumer function is running |
| `thread_active` | Gauge | `thread_slot` | `1` while executing a job under a lease; `0` is idle/non-processing |
| `thread_last_successful_poll_timestamp_seconds` | Gauge | `thread_slot` | Unix timestamp of the last completed claim query, including an empty result |
| `thread_active_job_duration_seconds` | Gauge | `thread_slot` | Live duration of the leased dispatch; `0` when inactive |
| `job_attempts_total` | Counter | `job_type`, `outcome` | Attempts with a confirmed durable outcome |
| `job_processing_duration_seconds` | Histogram | `job_type`, `outcome` | Leased dispatch duration; buckets span 0.1 seconds to 1 hour plus `+Inf` |
| `job_deferrals_total` | Counter | `job_type`, `reason` | Durable requeues that restore the attempt budget |
| `stale_jobs_recovered_total` | Counter | `outcome` | Stale RUNNING jobs moved durably to retry or dead letter |
| `poll_errors_total` | Counter | `thread_slot` | Failed claim queries or idle database/schema checks |
| `job_operation_errors_total` | Counter | `job_type`, `operation` | Dispatch/deferral operations without a confirmed durable outcome |
| `lease_failures_total` | Counter | `thread_slot`, `phase` | Unexpected lease acquisition or heartbeat-connection failures |
| `queue_ready_jobs` | Gauge | none | PENDING/FAILED jobs eligible to claim now |
| `queue_oldest_ready_wait_seconds` | Gauge | none | Time since the oldest ready job's `available_at` |
| `queue_running_jobs` | Gauge | none | Current RUNNING rows |
| `queue_dead_letter_jobs` | Gauge | none | Current DEAD rows |
| `queue_collection_success` | Gauge | none | `1` if the latest snapshot query succeeded, otherwise `0` |
| `queue_last_success_timestamp_seconds` | Gauge | none | Unix timestamp of the last successful snapshot |
| `queue_collection_errors_total` | Counter | none | Failed snapshot queries |

Prefix every short name in the table with `adept_engine_worker_`. Prometheus
histograms additionally export `_bucket`, `_sum` and `_count` series.

Bounded label values are:

- Attempt outcomes: `succeeded`, `retry_scheduled`, `dead_lettered`.
- Processing-only outcomes: the attempt outcomes plus
  `continuation_requeued` and `operational_error`.
- Deferral reasons: `continuation`, `scope_busy`, `shutdown`.
- Job operations: `dispatch`, `deferral`.
- Lease phases: `acquire`, `connection_lost`.
- Stale-recovery outcomes: `retry_scheduled`, `dead_lettered`.

No metric contains a job UUID, worker owner ID, repository/workspace ID, email,
URL, exception text or other unbounded/customer-specific value.

## Semantics and failure behavior

- Monitoring startup is best effort: a port-bind or sampler-start failure logs
  `engine_worker_monitoring_start_failed` with the component and error type,
  but processing threads still start. Each component starts independently;
  monitoring shutdown errors also cannot skip the remaining worker cleanup.
  There is no automatic startup retry: fix the configuration/resource issue and
  restart the worker. PR 3 must alert on an unreachable scrape target and stale
  queue snapshots; a live endpoint alone does not prove monitoring is healthy.
- An empty queue is healthy idle behavior. A successful empty snapshot reports
  ready count and oldest wait as zero.
- Ready work exactly matches claiming: status PENDING or FAILED,
  `available_at <= now()`, and remaining attempts. Future schedules and retry
  backoff are excluded. Waiting time starts at `available_at`, not `created_at`.
- `retry_scheduled` means a failed attempt was durably stored as FAILED for a
  later retry. `dead_lettered` means the durable result is DEAD.
- Pagination/continuation, a busy repository/workspace scope, and shutdown
  hand-back are deferrals. They restore the claimed attempt and are not counted
  as ordinary failures or retry attempts.
- An `operational_error` duration means dispatch returned without a confirmed
  durable result, such as lost ownership. It increments the bounded job
  operation error counter, not an attempt outcome.
- Queue state is queried once every 60 seconds by a separate sampler, never by
  every worker poll or HTTP scrape. The single read-only query has a five-second
  PostgreSQL statement timeout and takes no row locks. It uses the existing
  claim index for ready work; exact RUNNING/DEAD counts need no migration but
  may justify an API-owned index later if production scale proves it necessary.
- Before the first successful snapshot, queue value gauges are `NaN`, success
  is `0`, and the last-success timestamp is `0`. On an error, last good values
  and timestamp remain unchanged, success becomes `0`, and the error counter
  increases. A database error therefore cannot look like an empty queue.
- Prometheus `up{job="adept-engine-worker"}` only proves that Alloy reached the
  endpoint. Determine worker health from expected thread liveness, recent
  successful polls, queue freshness and active-job duration together.
- During a long job, `thread_active` stays `1` and its duration keeps increasing;
  the poll timestamp can legitimately age until that job finishes. During
  healthy idle time, the thread stays alive/ inactive and keeps polling.

Counters and histogram state are in memory and reset when the worker process
restarts. Every bounded counter/histogram label combination, including `UNKNOWN`,
is published at zero before its first event. This gives `rate`/`increase` a
baseline when a scrape precedes that event; events before the first scrape or
between a final scrape and a restart can still be missed. Initialization does
not record fake jobs or duration samples. Thread series only use configured
slots, and queue values remain `NaN` until their first successful snapshot.
Queue gauges rebuild from durable PostgreSQL state on the next successful
sample. A heartbeat connection loss still terminates the process immediately;
its in-memory counter may disappear before a scrape, while the later durable
stale-recovery counter records the resulting retry/dead-letter transition.

SIGTERM/SIGINT sets `shutdown_requested`, stops new claims and leaves the
endpoint/sampler running while active handlers drain. The sampler and endpoint
stop only after the processing threads finish. The existing 120-second Docker
deployment grace period and stale-claim recovery remain unchanged.

## PR 3 Alloy and Compose handoff

In `adept-api`, PR 3 should add the worker's private Compose settings without a
host port:

```yaml
engine-worker:
  environment:
    ENGINE_WORKER_THREADS: "${ENGINE_WORKER_THREADS:-2}"
    ENGINE_METRICS_BIND_ADDRESS: "0.0.0.0"
    ENGINE_METRICS_PORT: "8001"
    ENGINE_QUEUE_METRICS_INTERVAL_SECONDS: "${ENGINE_QUEUE_METRICS_INTERVAL_SECONDS:-60}"
  expose:
    - "8001"
```

Add `ENGINE_WORKER_THREADS=2` and
`ENGINE_QUEUE_METRICS_INTERVAL_SECONDS=60` to the non-secret production example.
The current default already runs two threads, but explicit Compose forwarding
is required for the documented one-thread rollback to work.

Extend PR 1's Alloy configuration with:

```alloy
prometheus.scrape "engine_worker_metrics" {
	sample_limit    = 5000
	targets         = [{__address__ = "engine-worker:8001"}]
	metrics_path    = "/metrics"
	forward_to      = [prometheus.relabel.engine_worker_metrics.receiver]
	scrape_interval = "60s"
	scrape_timeout  = "15s"
	job_name        = "adept-engine-worker"
}

prometheus.relabel "engine_worker_metrics" {
	forward_to = [prometheus.remote_write.grafana_cloud.receiver]

	rule {
		source_labels = ["__name__"]
		regex         = "up|scrape_.*|adept_engine_worker_.*"
		action        = "keep"
	}
}
```

Alloy and `engine-worker` are already on the private `adept` network from PR 1.
Do not add Grafana variables to the worker, publish port 8001, route it through
Caddy or use this endpoint as the worker's Docker health check.

PR 2 itself requires no AWS file installation and does not access AWS. After PR
2 is reviewed and merged, the existing engine workflow builds/deploys the image
and restarts the existing Engine API and single worker container. Monitoring
does not begin until the production owner reviews and installs PR 3's updated
`adept-api` Compose/Alloy files using the PR 1 runbook.

After PR 3 installation, validate without printing secrets:

```bash
dc --profile monitoring config --quiet
dc exec -T caddy curl -fsS -o /dev/null http://engine-worker:8001/metrics
```

Then verify fresh `up{job="adept-engine-worker"}` and
`adept_engine_worker_configured_threads{job="adept-engine-worker"}` samples in
Grafana Cloud. Roll back monitoring by removing the Alloy worker scrape and
recreating only Alloy; roll back worker code with the prior immutable engine
image. Neither action changes PostgreSQL data or Grafana credentials.
