# Available CI Benchmark Options

This file is the authoritative guide for choosing which benchmark command, if any, should be suggested for a pull request.

The workflow reads the YAML block below.

- Rules are evaluated in order, top to bottom.
- The first matching rule wins.
- Put the most specific rules first.
- Use `command: null` for "no suggestion".

```yaml
rules:
  - command: /benchmark extended
    reason: Changes under `src/kd_tree/` can affect within/nearest-within query performance and should use the extended query-family benchmark suite.
    match_any:
      - src/kd_tree/**

  - command: /benchmark
    reason: Changes under `src/` or to `Cargo.toml` can affect core runtime performance and should use the basic benchmark suite.
    match_any:
      - src/**
      - Cargo.toml

  - command: null
    reason: No benchmark suggestion is needed when none of the higher-priority rules match.
    match_any:
      - "**"
```

## Notes

- `/benchmark extended` is reserved for query-pipeline changes and should stay above the broader `/benchmark` rule.
- `/benchmark` is the default command for the basic benchmark suite and for general source changes.
