# apps/backend/app/metrics.py
from __future__ import annotations
import time
from collections import defaultdict

# Counters
requests_total = defaultdict(int)      # key: (route, method, status)
errors_total = defaultdict(int)        # key: (route, status)

# Timings (simple histogram buckets in ms)
# Prometheus-like buckets: 50, 100, 200, 400, 800, 1600, +inf
BUCKETS = [50, 100, 200, 400, 800, 1600]
request_duration = defaultdict(lambda: [0]*(len(BUCKETS)+1))  # key: route -> counts per bucket
inference_duration = defaultdict(lambda: [0]*(len(BUCKETS)+1))  # key: model_version

def now_ms() -> int:
    return int(time.time() * 1000)

def observe(hist, key, ms: int):
    arr = hist[key]
    for i, b in enumerate(BUCKETS):
        if ms <= b:
            arr[i] += 1
            break
    else:
        arr[-1] += 1  # +inf

def render_prometheus() -> str:
    lines = []
    # HELP/TYPE
    lines += [
        "# HELP VisionScale_requests_total Total HTTP requests by route/method/status.",
        "# TYPE VisionScale_requests_total counter",
    ]
    for (route, method, status), val in requests_total.items():
        lines.append(f'VisionScale_requests_total{{route="{route}",method="{method}",status="{status}"}} {val}')

    lines += [
        "# HELP VisionScale_errors_total Total error responses by route/status.",
        "# TYPE VisionScale_errors_total counter",
    ]
    for (route, status), val in errors_total.items():
        lines.append(f'VisionScale_errors_total{{route="{route}",status="{status}"}} {val}')

    lines += [
        "# HELP VisionScale_request_duration_ms Request duration histogram (ms).",
        "# TYPE VisionScale_request_duration_ms histogram",
    ]
    for route, buckets in request_duration.items():
        cumulative = 0
        for i, b in enumerate(BUCKETS):
            cumulative += buckets[i]
            lines.append(f'VisionScale_request_duration_ms_bucket{{route="{route}",le="{b}"}} {cumulative}')
        cumulative += buckets[-1]
        lines.append(f'VisionScale_request_duration_ms_bucket{{route="{route}",le="+Inf"}} {cumulative}')
        # sum & count (approximate: use midpoints except +Inf omitted)
        count = sum(buckets)
        sum_ms = 0
        mids = [25, 75, 150, 300, 600, 1200]  # midpoints for our buckets
        for i, n in enumerate(buckets[:-1]):
            sum_ms += n * mids[i]
        # +Inf bucket: just add boundary * count (very rough)
        sum_ms += buckets[-1] * 1600
        lines.append(f'VisionScale_request_duration_ms_sum{{route="{route}"}} {sum_ms}')
        lines.append(f'VisionScale_request_duration_ms_count{{route="{route}"}} {count}')

    lines += [
        "# HELP VisionScale_inference_duration_ms Inference duration histogram (ms).",
        "# TYPE VisionScale_inference_duration_ms histogram",
    ]
    for model, buckets in inference_duration.items():
        cumulative = 0
        for i, b in enumerate(BUCKETS):
            cumulative += buckets[i]
            lines.append(f'VisionScale_inference_duration_ms_bucket{{model="{model}",le="{b}"}} {cumulative}')
        cumulative += buckets[-1]
        lines.append(f'VisionScale_inference_duration_ms_bucket{{model="{model}",le="+Inf"}} {cumulative}')
        count = sum(buckets)
        # rough sum same as above
        mids = [25, 75, 150, 300, 600, 1200]
        sum_ms = sum(buckets[i] * mids[i] for i in range(len(BUCKETS)))
        sum_ms += buckets[-1] * 1600
        lines.append(f'VisionScale_inference_duration_ms_sum{{model="{model}"}} {sum_ms}')
        lines.append(f'VisionScale_inference_duration_ms_count{{model="{model}"}} {count}')

    return "\n".join(lines) + "\n"