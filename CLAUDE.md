# Support Agent — StuntDouble Sample

Sample customer support agent demonstrating StuntDouble mocking.

## Commands

```bash
uv sync          # Install deps (StuntDouble from its public Git repository)
uv run pytest    # Run all tests
```

## Structure

- `src/support_agent/` — agent code (tools, graph, prompts)
- `scenarios/` — JSON mock scenario files
- `tests/` — pytest tests demonstrating StuntDouble patterns
