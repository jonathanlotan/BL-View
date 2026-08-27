What’s there

One command: ./run.sh creates the venv, synthesises 24 h of raw CL31 data, runs ingest → preprocess → detect → track → store, asserts the injected layers were recovered, and serves the quicklook. ./run.sh --test runs 101 unit tests plus the validation harness.

No sample file existed in the repo, so everything is built and validated against generated data. The synthetic file mimics real CL31 message structure (20-bit two’s-complement hex profiles, status/parameter lines, CRC-16) and carries a sidecar recording exactly what was injected at each timestamp.

Measured across seven synthetic days (varying diurnal phase and noise seed — single-run figures aren’t a result here):

feature	detection rate	within tol	MAE	max FP
cloud base	98.3–100 %	100 %	24.0–24.6 m	8.9 %
mixing layer top	92.2–98.4 %	89.9–91.5 %	37.7–45.7 m	0.0 %
residual layer top	84.4–86.1 %	99.2–100 %	17.0–22.7 m	0.0 %
haze base / top	90.2–96.0 %	100 %	9.0–19.9 m	5.4 %

Fog and precipitation screening: 100 % recall at ~99 % precision.

Four things worth flagging

Textbook Brooks 2003 wasn’t enough, and each gap showed up by measuring against truth rather than by inspection. Edges are found in log space (linear lets the near-surface gradient dwarf every elevated edge) but localised in linear space, since log biases tops upward. An absolute floor on edge amplitude was needed alongside statistical significance — inside a mixing layer the SNR reaches several hundred, so a 1 % ripple is significant but meaningless; without the floor, p95 mixing-layer error was 925 m rather than ~300 m.

Two correlated systematics are propagated as bias, not noise — the overlap-model residual and the log-clamp ramp where signal drops below the R²-growing noise floor. Summing per-gate variances lets a correlated error average away and makes an artefact look highly significant; both reliably produced phantom layers.

A real concurrency bug, found by opening the page rather than by testing: netCDF4/HDF5 isn’t thread-safe and FastAPI runs sync handlers in a threadpool, so one browser load killed the server process. Now serialised, with a regression test.

Two limitations I did not tune away. Where the mixing and residual layers genuinely merge (the ~09:00–10:30 and 18:00–19:00 transitions, aerosol contrast falling to ~1.14), BL View reports one merged layer rather than inventing a boundary — that accounts for essentially all remaining mixing-layer error, and it reports it with a confidence that doesn’t yet reflect the ambiguity. And nothing is reported below 90 m: since the real overlap function isn’t published, gates needing more than 8× amplification are masked rather than fabricated. Both are in the README’s limitations, along with light drizzle/virga escaping the precipitation screen — the most likely source of a wrong layer on real data.

71 numbered assumptions are in ASSUMPTIONS.md, grouped by stage and cross-referenced from the code. The #G1 framing holds throughout: every height is an aerosol backscatter gradient, stated in the UI banner, the API payloads, the netCDF attributes and the CLI help.