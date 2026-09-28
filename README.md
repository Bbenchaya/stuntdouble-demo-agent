# StuntDouble Sample Agent

A sample customer support agent demonstrating [StuntDouble](https://github.com/intuit/stunt-double) mocking features.

## Quick Start

```bash
uv sync
uv run pytest
```

### Python mock factories

Tool mocks are defined in `src/support_agent/mocked_tools.py` (scenario JSON under `mocks` drives outputs). The CLI
registers them with signature validation.

### Live mock visualization

Stream all mocked workflows, including tool arguments, resolved mock results, and final agent responses:

```bash
uv run python scripts/visualize_mock_run.py
```

The visualizer uses terminal colors when supported and clearly identifies the run as offline. It exercises the real
LangGraph workflow with a scripted local chat model and StuntDouble tool mocks; it does not call an external LLM or real
tool backends.

Run one workflow without presentation delays:

```bash
uv run python scripts/visualize_mock_run.py --scenario payment --delay 0
```

Force or disable ANSI colors when redirecting output:

```bash
uv run python scripts/visualize_mock_run.py --color always
uv run python scripts/visualize_mock_run.py --color never
```

### Commit runtime regression harness

Run the regression checks against the currently installed StuntDouble Git revision:

```bash
uv run python scripts/verify_stuntdouble_commit.py
```

Pass any full or abbreviated commit hash to test that exact revision:

```bash
uv run python scripts/verify_stuntdouble_commit.py --commit <7-to-40-character-commit-hash>
```

If the requested revision is not already installed, the script uses `uv` to create an isolated environment from that
Git commit; it does not modify this project's environment or dependency pin. A network connection is only needed to
fetch a commit that is not in the local `uv` cache. The runtime checks do not call any external service.

The harness launches `scripts/local_mcp_test_server.py` as a real stdio subprocess and verifies:

- the requested StuntDouble Git revision is installed;
- `returns_fn()` honors `.when(...)` and `.echoes_input(...)`;
- structured `AIMessage` text blocks are parsed correctly;
- an unresponsive MCP server times out, cleans up, and permits a retry;
- a noisy MCP server cannot deadlock the client, and notifications, stale response IDs, and stdout logs are ignored.

During MCP checks, the output streams server startup commands, process IDs, handshake status, discovered tools and required
inputs, stderr-drain statistics, tool calls, results, and shutdown.

Run an individual regression area when diagnosing a failure:

```bash
uv run python scripts/verify_stuntdouble_commit.py --commit <hash> --only mcp-timeout
uv run python scripts/verify_stuntdouble_commit.py --commit <hash> --only mcp-noise --color always
```

## What This Demonstrates

| Feature | File |
|---------|------|
| Python mock factories + JSON | `src/support_agent/mocked_tools.py` |
| Fluent builder API | `tests/test_check_bills.py` |
| Data-driven JSON scenarios | `tests/test_make_payment.py` |
| Mixed patterns | `tests/test_update_profile.py` |
| Call recording & assertions | All test files |
| Input matching operators | `scenarios/*.json` |
| Dynamic placeholders | `scenarios/*.json` |

## Agent Workflows

- **Check Bills** — look up customer, list overdue bills
- **Make Payment** — look up bill, submit payment
- **Update Profile** — get profile, update fields

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         python -m support_agent                         │
│                                                                         │
│   args → workflow (check_bills | make_payment | update_profile)         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
           ┌───────────────────────────────────────┐
           │   Load scenario JSON                  │
           │   scenarios/<workflow>.json           │
           │   ─────────────────────────           │
           │   { mocks: { tool_name: [cases] } }   │
           └──────────────────┬────────────────────┘
                              │ inject_scenario_metadata
                              ▼
           ┌──────────────────────────────────────────┐
           │   RunnableConfig  (carries scenario)     │
           └──────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      build_graph(llm, wrapper)                          │
│                                                                         │
│                                                                         │
│     ┌───────┐      ┌───────────────┐        ┌──────────────────────┐    │
│     │ START │ ───▶ │  agent node   │ ─────▶ │  tools_condition     │    │
│     └───────┘      │  (llm+tools)  │        └─────┬────────────┬───┘    │
│                    └───────▲───────┘              │            │        │
│                            │                  tool_calls?     END       │
│                            │                      │                     │
│                            │                      ▼                     │
│                            │           ┌──────────────────────┐         │
│                            └───────────│      ToolNode        │         │
│                                        │ awrap_tool_call =    │         │
│                                        │   StuntDouble wrapper│         │
│                                        └──────────┬───────────┘         │
│                                                   │                     │
└───────────────────────────────────────────────────┼─────────────────────┘
                                                    │
                                                    ▼
                 ┌──────────────────────────────────────────────────┐
                 │          StuntDouble interception                │
                 │                                                  │
                 │   ┌────────────────┐    ┌──────────────────┐     │
                 │   │ MockToolsReg   │    │  CallRecorder    │     │
                 │   │ (tool→factory) │    │  (args/result/   │     │
                 │   │                │    │   was_mocked)    │     │
                 │   └───────┬────────┘    └──────────────────┘     │
                 │           │                                      │
                 │           ▼                                      │
                 │   mocked_tools.py factories                      │
                 │           │                                      │
                 │           ▼                                      │
                 │   InputMatcher(case.input, kwargs)               │
                 │   resolve_output(case.output, kwargs)            │
                 │   ─ placeholders: {{sequence('BILL')}},          │
                 │                   {{now - 14d}}                  │
                 └──────────────────────┬───────────────────────────┘
                                        │ mocked dict result
                                        ▼
                               back to agent node
                               (loop until no tool_calls)


LLM source:

  ┌────────────── offline path ─────────────┐
  │ FakeChatWithTools                       │
  │  ├─ scripted AIMessage list             │
  │  └─ bind_tools = no-op                  │
  └─────────────────────────────────────────┘
```
