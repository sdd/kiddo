# Available CI Benchmark Options

This file is the authoritative guide for choosing which benchmark command, if any, should be suggested for a pull request.

Prefer no suggestion unless the changed code has a plausible performance impact.

## `/benchmark`

Suggest `/benchmark` when the change is most likely to affect the main query and traversal hot paths, especially:

- tree traversal logic
- nearest-neighbour query logic
- branch layout or cache-locality-sensitive code
- Eytzinger-specific code
- benchmark harness changes that primarily affect the Eytzinger-focused benchmark suite
- other performance-sensitive changes where the Eytzinger-focused suite is the closest fit

## `/benchmark dist`

Suggest `/benchmark dist` when the change is most likely to affect distance metric computation or pruning behavior driven by the metric, especially:

- code under `src/dist/`
- metric-specific pruning logic
- ISA-specific distance implementations
- metric widening, accumulation, or comparison changes
- benchmark harness changes that primarily affect the distance-metrics benchmark suite

## No Benchmark Suggestion

Do not suggest a benchmark command for changes that are unlikely to affect runtime performance, for example:

- documentation
- changelog updates
- release automation
- formatting-only changes
- CI-only changes unrelated to benchmark execution
- refactors that do not change hot-path behavior
- tests added around unchanged implementation code

## Decision Rules

- If the change plausibly affects both categories, choose the command that best matches the dominant risk area.
- If the impact is broad but primarily query/traversal-oriented, prefer `/benchmark`.
- If the impact is speculative or weak, prefer no suggestion.
- The workflow should suggest at most one benchmark command.
