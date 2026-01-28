# Project: Multi-Agent Software Development Framework (Python, Linux)

## 1. Goal and Scope

Implement a **multi-agent software development framework** in Python for **Linux**.

High-level idea:

* A user defines **agents with roles** (e.g. “Architect”, “Coder”, “Reviewer”, “Tester”), each backed by an LLM.
* The framework **orchestrates** these agents so they can:

  * Read and understand a codebase.
  * Propose and apply code changes.
  * Build / compile the project.
  * Run tests or executables.
* The framework must:

  * Support **multiple LLM providers** via a pluggable API layer.
  * Run all code execution in a **controlled environment** (preferably Docker) to keep the host safe.
  * Be usable as a **Python library** and via a **CLI**.
  * Be structured so that additional agents and backends can be added without touching core logic.

Non-goals (for the initial implementation):

* No GUI / web frontend required (CLI only).
* No distributed cluster orchestration; a single host process is enough.
* No persistent database; file-based or in-memory state is sufficient for v1.

Target Python: **Python 3.11+**.
Target OS: **Linux (x86-64)**.

---

## 2. Overall Architecture

Implement a modular architecture with these main components:

1. **Core Orchestrator**

   * Central coordinator for all agents, tasks, and events.
   * Maintains a registry of agents and their roles.
   * Routes messages between agents.
   * Coordinates access to:

     * LLM backend layer
     * Workspace / file access
     * Code execution engine
     * Memory / knowledge base

2. **Agent Framework**

   * Base `Agent` abstraction:

     * Has an ID, a role definition, and configuration (LLM model, tools).
     * Receives messages, decides actions (LLM calls, tools, delegation).
     * Sends messages back to orchestrator.
   * Concrete agent types for v1:

     * `PlannerAgent` – breaks down user task into steps/subtasks.
     * `CodingAgent` – reads files, proposes edits, writes code.
     * `ReviewerAgent` – reviews diffs, comments, requests changes.
     * `TesterAgent` – invokes build + tests via execution engine and summarizes results.
   * Agents should be **stateless in storage** but may hold in-memory conversation/history while a session is running.

3. **LLM Backend Layer**

   * Abstraction over different LLM APIs.
   * Provide a common `LLMClient` interface:

     * Methods: `complete_chat(messages, **options)` and possibly `complete_text(prompt, **options)`.
   * Implement backends as pluggable classes:

     * `OpenAIClient` (OpenAI-compatible HTTP API).
     * `AzureOpenAIClient` (if needed).
     * `GenericOpenAICompatibleClient` (base for others).
   * Load API keys and endpoints from configuration (env vars and/or config files).
   * Support **rate limiting / basic retry**.

4. **Workspace & File Manager**

   * Abstraction for working with a **local codebase directory**.
   * Core requirements:

     * Safe, explicit root directory; no access above repo root.
     * APIs to:

       * List files (with filters, e.g. only `*.py`, `*.cpp`).
       * Read file content (full or line ranges).
       * Write/overwrite files.
       * Create new files and directories.
       * Compute simple diffs between “before” and “after”.
     * Maintain an in-memory “staging” view vs. on-disk state (optional but nice-to-have) so agents can preview changes.
   * Should be usable both by agents and by the orchestrator.

5. **Code Execution Engine**

   * Component that safely executes commands related to the target codebase:

     * Build / compile commands (e.g. `cmake`, `make`, `ninja`, `pytest`, `ctest`, etc.).
     * Arbitrary shell commands needed for experiments and debugging.
   * Provide a generic `ExecutionResult` object (stdout, stderr, exit code, duration, optional logs).
   * Two execution modes (configurable):

     1. **Docker sandbox**:

        * Run commands inside a container based on a configurable image (e.g. `python:3.11-slim`).
        * Bind-mount the workspace into the container (read/write).
        * Allow setting environment variables.
     2. **Local runner**:

        * Run commands directly on the host (for development only).
   * Execution API must be simple enough for LLM-generated flows:

     * `run(command: list[str], timeout: int | None, cwd: Path | None) -> ExecutionResult`

6. **Memory / Knowledge Base**

   * Simple in-memory and/or file-based memory for:

     * Conversation history per session.
     * Short summaries of large files or recent code changes.
   * Provide a **vector-store-like interface** for future extension, but v1 can use naive in-memory lists and embeddings stubbed out.
   * Memory is an internal service accessible by agents via the orchestrator.

7. **Configuration System**

   * Single configuration model that can be loaded from:

     * `pyproject.toml` or `config.yaml` at repo root.
     * Environment variables for secrets (API keys).
   * Configurable aspects:

     * LLM providers, models, and API keys.
     * Default workspace path.
     * Default execution mode (docker/local).
     * Agent definitions and roles.

8. **User Interface & CLI**

   * Provide a `multiagent-dev` CLI entry point.
   * Core commands:

     * `multiagent-dev init` – initialize configuration for a repository.
     * `multiagent-dev run-task` – run a high-level task through the multi-agent workflow.
     * `multiagent-dev plan` – only run planning / architecture steps, no code writes.
     * `multiagent-dev exec` – execute a specific agent (e.g. only tests).
   * CLI options should include:

     * Path to workspace.
     * Agent set / profile to use.
     * Whether to allow file writes.
     * Whether to allow code execution, and with which mode (docker/local).

---

## 3. Directory Structure

Implement the project as a Python package with this structure:

```text
multiagent-dev/
├─ pyproject.toml           # Use hatchling or poetry; pick one and configure.
├─ README.md
├─ PROJECT_SPEC.md          # This spec (or README can include it)
├─ src/
│  └─ multiagent_dev/
│     ├─ __init__.py
│     ├─ config.py          # Configuration loading & models
│     ├─ orchestrator.py    # Core Orchestrator implementation
│     ├─ agents/
│     │  ├─ __init__.py
│     │  ├─ base.py         # Base Agent class and message types
│     │  ├─ planner.py
│     │  ├─ coder.py
│     │  ├─ reviewer.py
│     │  └─ tester.py
│     ├─ llm/
│     │  ├─ __init__.py
│     │  ├─ base.py         # LLMClient interface
│     │  ├─ openai_client.py
│     │  └─ registry.py     # Factory to construct LLM clients from config
│     ├─ workspace/
│     │  ├─ __init__.py
│     │  └─ manager.py      # File and project operations
│     ├─ execution/
│     │  ├─ __init__.py
│     │  ├─ base.py         # Executor base
│     │  ├─ docker_exec.py
│     │  └─ local_exec.py
│     ├─ memory/
│     │  ├─ __init__.py
│     │  └─ memory.py       # Simple conversation & document memory
│     ├─ cli/
│     │  ├─ __init__.py
│     │  └─ main.py         # CLI entrypoints (typer or argparse)
│     └─ util/
│        ├─ __init__.py
│        └─ logging.py      # Logging helpers
└─ tests/
   └─ ...                   # Unit tests
```

The structure can evolve, but keep modules and responsibilities similar.

---

## 4. Detailed Component Specifications

### 4.1 Orchestrator

Create a class `Orchestrator` in `orchestrator.py` with responsibilities:

* Hold:

  * Mapping `agent_id -> Agent` instance.
  * Global configuration object.
  * References to workspace manager, execution engine, memory service, and LLM factory.
* Expose methods:

  * `register_agent(agent: Agent) -> None`
  * `get_agent(agent_id: str) -> Agent | None`
  * `send_message(to_agent: str, message: AgentMessage) -> None`
  * `run_task(task: UserTask) -> TaskResult`
* Implement a simple **event loop**:

  * Synchronous is fine for v1 (no need for full async).
  * Use a queue of messages (FIFO).
  * Each message includes `sender`, `recipient`, and `payload`.
* The orchestrator:

  * Starts with an initial `UserTask` created from the CLI.
  * Publishes this to `PlannerAgent`.
  * Then processes messages until a termination condition is met (task completed or failed).

Define message types in `agents/base.py`:

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

Define `UserTask` and `TaskResult` types in `orchestrator.py` or a shared `types.py`.

### 4.2 Agent Base Class

In `agents/base.py` implement:

```python
class Agent(ABC):
    def __init__(
        self,
        agent_id: str,
        role: str,
        llm_client: LLMClient,
        orchestrator: "Orchestrator",
        workspace: WorkspaceManager,
        executor: CodeExecutor,
        memory: MemoryService,
    ): ...

    @abstractmethod
    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Process a message and return new messages to send."""
```

Agents should:

* Use the `llm_client` to generate responses.
* Encode their internal instructions (role description) as a **system prompt** for the LLM.
* Be able to:

  * Read files via `workspace`.
  * Request code execution via `executor`.
  * Store and retrieve notes via `memory`.

### 4.3 Built-in Agents

Implement the following agents with minimal but functional behavior:

1. **PlannerAgent**

   * Role: Take the user’s high-level task (e.g. “Add a feature X”).
   * Responsibilities:

     * Use LLM to break down task into a list of steps.
     * Send messages to other agents:

       * `CodingAgent` for implementation steps.
       * `TesterAgent` for testing steps.
       * `ReviewerAgent` when review is needed.
   * Implementation notes:

     * Prompt LLM with the repository description (optional) and user task.
     * Produce a lightweight step plan stored in `memory`.
     * Generate explicit messages with clear `content` that other agents can follow.

2. **CodingAgent**

   * Role: Make code changes requested by Planner or Reviewer.
   * Responsibilities:

     * Read relevant files using `workspace`.
     * Propose modifications.
     * Write code back to disk.
   * Implementation notes:

     * Use an inner prompting scheme:

       * Provide file contents and requested change.
       * Ask LLM to return only the changed file(s) in a structured format (e.g. JSON or unified patch).
     * For v1, accept a simple protocol:

       * LLM outputs full new file content between markers like `FILE: path`, `CODE: ...`.
       * Parse and apply using `workspace`.

3. **ReviewerAgent**

   * Role: Review changes and request improvements.
   * Responsibilities:

     * Use `workspace` to inspect modified files.
     * Possibly run `pytest` or `ctest` via `executor`.
     * Send messages back to `CodingAgent` with comments or approval.
   * Implementation notes:

     * For v1, implement review as:

       * Compare last modified files with previous contents (if available).
       * Ask LLM: “Review diff; list issues and improvements”.
     * Use acceptance / rejection metadata in `AgentMessage.metadata`.

4. **TesterAgent**

   * Role: Run tests and summarize results.
   * Responsibilities:

     * Run configured build / test commands via `executor`.
     * Summarize results and send them to Planner / Reviewer.
   * Implementation notes:

     * Support configuring test commands in config (e.g. `["pytest", "-q"]`).
     * Provide logs and exit codes to the LLM when summarizing.

### 4.4 LLM Layer

Define `LLMClient` in `llm/base.py`:

```python
class LLMClient(ABC):
    @abstractmethod
    def complete_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...
```

* `messages` are dicts like `{"role": "system"|"user"|"assistant", "content": "..."}`.

Provide an `OpenAIClient` implementing `LLMClient` using a standard OpenAI-compatible HTTP API:

* Configurable endpoint, model name, and API key.
* Use synchronous requests with reasonable timeout and simple retries.

Implement a `registry` that creates the right client from configuration:

```python
def create_llm_client(config: LLMConfig) -> LLMClient: ...
```

### 4.5 Workspace Manager

In `workspace/manager.py`:

* Represent the workspace root as `Path`.
* Provide methods:

```python
class WorkspaceManager:
    def __init__(self, root: Path): ...

    def list_files(self, pattern: str | None = None) -> list[Path]: ...
    def read_text(self, path: Path) -> str: ...
    def write_text(self, path: Path, content: str) -> None: ...
    def file_exists(self, path: Path) -> bool: ...
    def compute_unified_diff(self, old: str, new: str, path: Path) -> str: ...
```

* Ensure that paths are always inside `root` (reject anything that escapes via `..`).

### 4.6 Execution Engine

In `execution/base.py`:

```python
@dataclass
class ExecutionResult:
    command: list[str]
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
```

Define:

```python
class CodeExecutor(ABC):
    @abstractmethod
    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...
```

Implement:

* `LocalExecutor` in `local_exec.py` using `subprocess.run`.
* `DockerExecutor` in `docker_exec.py` using `subprocess` to invoke `docker run` with:

  * Configurable image.
  * Bind-mounted workspace.
  * Non-root user if possible.

### 4.7 Memory Service

In `memory/memory.py` implement a simple class:

```python
class MemoryService:
    def __init__(self):
        self._conversations: dict[str, list[AgentMessage]] = {}
        self._notes: dict[str, str] = {}

    def append_message(self, session_id: str, message: AgentMessage) -> None: ...
    def get_messages(self, session_id: str) -> list[AgentMessage]: ...
    def save_note(self, key: str, text: str) -> None: ...
    def get_note(self, key: str) -> str | None: ...
```

This is enough for v1; it can later be extended to vector search.

### 4.8 Configuration

In `config.py` create Pydantic models or dataclasses:

* `LLMConfig`
* `ExecutorConfig`
* `AgentConfig`
* `AppConfig` (top-level)

Support loading:

* From `pyproject.toml` section `[tool.multiagent_dev]`, or
* From `multiagent_dev.yaml`.

Provide:

```python
def load_config(path: Path | None = None) -> AppConfig: ...
```

If no config file exists, use reasonable defaults and allow CLI to generate one.

### 4.9 CLI

Use either `typer` or `argparse`. Example commands (typer preferred):

```python
@app.command()
def init(workspace: Path = Path(".")):
    """Initialize config in the given workspace."""
    ...

@app.command("run-task")
def run_task(
    description: str,
    workspace: Path = Path("."),
    allow_write: bool = True,
    execution_mode: str = typer.Option("local", "--exec-mode", help="local|docker"),
):
    """Run a multi-agent workflow to handle the described task."""
    ...
```

Entry point in `pyproject.toml`:

```toml
[project.scripts]
multiagent-dev = "multiagent_dev.cli.main:app"
```

---

## 5. Implementation Phases for the Code Model

Implement the project in the following order. After each phase, run tests and ensure code passes `ruff`/`flake8` and `mypy` or `pyright` where configured.

### Phase 1 – Project Bootstrap

1. Create `pyproject.toml` with:

   * Package name `multiagent-dev`.
   * Dependencies:

     * `requests` (for HTTP to LLM).
     * `typer[all]` or `click` (for CLI).
     * `pydantic` or standard `dataclasses`.
     * Optional dev dependencies: `pytest`, `mypy`, `ruff`.
2. Create package structure in `src/multiagent_dev`.
3. Add basic `AppConfig` with minimal fields.
4. Add skeleton classes and empty tests.

### Phase 2 – Core Abstractions

1. Implement `AgentMessage`, `Agent` base class.
2. Implement `Orchestrator` with a simple synchronous message loop.
3. Implement `MemoryService`.
4. Add unit tests:

   * Message routing.
   * Simple agent stub responding to messages.

### Phase 3 – Workspace Manager

1. Implement safe file operations.
2. Add tests for path normalization and security (no path escape).

### Phase 4 – Execution Engine

1. Implement `LocalExecutor` with tests (mock subprocess).
2. Implement `DockerExecutor` skeleton (no heavy tests; just check command construction).

### Phase 5 – LLM Layer

1. Implement `LLMClient` interface and `OpenAIClient`.
2. Respect environment variables like `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.
3. Add minimal tests by mocking HTTP requests.

### Phase 6 – Agents

1. Implement `PlannerAgent` with a prompt template.
2. Implement `CodingAgent` with file-update protocol.
3. Implement `ReviewerAgent` and `TesterAgent`.
4. Add tests using fake `LLMClient` and fake `CodeExecutor` (no real API calls).

### Phase 7 – CLI

1. Wire CLI commands to orchestrator:

   * `init`
   * `run-task`
2. Use config loader and builder functions.

### Phase 8 – Polishing

1. Improve logging and error handling.
2. Add `README.md` with usage examples:

   * How to configure OpenAI API.
   * How to run a simple task on a sample repo.
3. Ensure all modules are type-checked and tested.

### Phase 9 – Tool & Function Registry

**Goal:**
Introduce first-class “tools” that agents can invoke in a structured, extensible way (beyond raw shell commands).

**Requirements:**

1. Define a `Tool` abstraction:

    * Name
    * Description
    * Input schema
    * Execution method
2. Implement a `ToolRegistry`:

    * Register tools by name.
    * Allow lookup by agents via orchestrator.
3. Built-in tools (minimum):

    * `run_command` (wraps CodeExecutor)
    * `read_file`
    * `write_file`
4. Agents must:

    * Invoke tools via orchestrator, not directly.
    * Never hard-code tool implementations.

**Definition of Done:**

* New tools can be added without modifying existing agents.
* Agents select tools via name + arguments.
* Tool execution is logged and testable.

---

### Phase 10 – Version Control (Git) Integration

**Goal:**
Allow agents to interact with Git in a controlled, reviewable way.

**Requirements:**

1. Introduce a `VersionControlService` abstraction.
2. Implement `GitService` with:

    * Status
    * Diff
    * Commit
    * Branch creation
3. Git operations must:

    * Be optional and configurable.
    * Never auto-push or create PRs by default.
4. ReviewerAgent must be able to:

    * Inspect diffs before approval.
    * Gate commits on approval.

**Definition of Done:**

* No agent executes `git` directly.
* Git access is sandboxed to the workspace.
* All commits are traceable to an agent decision.

---

### Phase 11 – Advanced Memory & Retrieval (RAG-Ready)

**Goal:**
Enable scalability to large codebases and long sessions.

**Requirements:**

1. Extend `MemoryService` to support:

    * Short-term (session) memory.
    * Long-term (project) memory.
2. Introduce a `RetrievalService` abstraction:

    * Interface only; implementation may be naive.
3. Support:

    * File summaries.
    * Chunked code storage.
    * Query-based retrieval.
4. Do **not** require a real vector DB in v1; in-memory is acceptable.

**Definition of Done:**

* Agents retrieve context via Memory/Retrieval services.
* No agent loads the entire repository into prompts.
* Retrieval logic is replaceable without agent changes.

---

### Phase 12 – Human-in-the-Loop Control

**Goal:**
Allow explicit user approvals and interventions at key points.

**Requirements:**

1. Introduce a `UserProxyAgent`:

    * Represents the human.
    * Can approve, reject, or modify decisions.
2. Add approval checkpoints:

    * Before code execution (optional).
    * Before committing changes.
3. Support execution modes:

    * Fully autonomous.
    * Approval-required.

**Definition of Done:**

* Orchestrator can pause and resume workflows.
* User input is explicit and logged.
* Autonomous and interactive modes share the same code paths.

---

### Phase 13 – Concurrency & Durable Execution

**Goal:**
Prepare the system for long-running and parallel workflows.

**Requirements:**

1. Refactor orchestrator to:

    * Support async execution (`asyncio`).
    * Allow parallel agent steps where safe.
2. Introduce workflow state persistence:

    * Serializable task state.
    * Resume after interruption.
3. Ensure determinism where required.

**Definition of Done:**

* Workflow can be paused and resumed.
* No race conditions between agents.
* Async code remains readable and testable.

---

### Phase 14 – Observability, Metrics, and Evaluation

**Goal:**
Make agent behavior measurable and debuggable.

**Requirements:**

1. Structured logging for:

    * Agent decisions.
    * LLM calls.
    * Tool executions.
2. Collect metrics:

    * Execution time.
    * Number of iterations.
    * Token usage (if available).
3. Add a basic evaluation harness:

    * Run predefined tasks.
    * Compare outcomes.

**Definition of Done:**

* Logs are machine-readable.
* Metrics can be exported.
* Failures are diagnosable post-mortem.

---

### Phase 15 – Multi-Language & Build Strategy Extensions

**Goal:**
Support heterogeneous projects more robustly.

**Requirements:**

1. Extend configuration to define:

    * Project language(s).
    * Build system(s).
    * Test commands per language.
2. Provide language profiles:

    * Python
    * C++ (CMake)
    * Generic shell
3. Execution logic must remain generic.

**Definition of Done:**

* No language-specific logic inside agents.
* Adding a new language requires configuration, not code changes.

---

## Final Note on Scope Control

Phases **1–8 define a complete, production-quality MVP**.
Phases **9–15 progressively evolve the system into a full-scale agentic development platform.

Each phase is **architecturally isolated** so that:

* Earlier phases never need redesign.
* Codex can stop after any phase with a coherent system.

---

## 6. Coding Guidelines for the AI Model

When implementing:

1. **Prefer explicit types** and docstrings for all public functions and classes.
2. Use **pure Python standard library** except where dependencies are stated.
3. Keep each module focused and small.
4. Add **unit tests** in `tests/` for each core component.
5. Avoid placeholder implementations like `pass` for public APIs; provide at least minimal, correct behavior.
6. Do not hard-code API keys or secrets; always read from environment or config.
7. Make sure the package is installable with `pip install .` from the repo root.

Below is an **additive section** you can append verbatim to the existing `PROJECT_SPEC.md`. It is written in a way that Codex (or any strong code model) will treat these requirements as **hard constraints**, not style suggestions.

You do **not** need to change any other sections; this integrates cleanly with the existing spec.

---

## 7. Code Quality, Architecture, and Engineering Standards (MANDATORY)

All code produced for this project **must** meet the following non-functional requirements. These are **hard requirements**, not guidelines.

### 7.1 Code Quality

The implementation must be:

* **Clean and readable**

  * Clear naming for variables, functions, classes, and modules.
  * No overly long functions; prefer small, single-purpose functions.
  * No deeply nested logic where simpler control flow is possible.
  * No “clever” code that trades readability for brevity.

* **Easy to maintain**

  * Changes in one component must not require changes in unrelated components.
  * Public APIs must be explicit and stable.
  * Internal details must be hidden behind well-defined interfaces.

* **Well-documented**

  * All public classes and functions must have docstrings explaining:

    * Purpose
    * Inputs
    * Outputs
    * Side effects (if any)
  * Non-obvious design decisions must be documented inline or in module docstrings.

### 7.2 SOLID Principles (STRICT)

The architecture must **explicitly follow the SOLID principles**:

1. **Single Responsibility Principle (SRP)**

   * Each class and module must have **one clear responsibility**.
   * Example:

     * Orchestrator coordinates agents only (no LLM logic, no file I/O).
     * WorkspaceManager handles files only (no LLM calls, no execution).

2. **Open/Closed Principle (OCP)**

   * Components must be **open for extension but closed for modification**.
   * New agent types, LLM providers, or executors must be addable without changing existing code.
   * Achieve this via:

     * Abstract base classes
     * Registries and factories
     * Dependency injection

3. **Liskov Substitution Principle (LSP)**

   * Any subclass must be safely usable wherever its base class is expected.
   * Example:

     * `DockerExecutor` must be usable anywhere `CodeExecutor` is used.
     * No subclass may weaken contracts or change expected behavior.

4. **Interface Segregation Principle (ISP)**

   * Interfaces must be **small and focused**.
   * Do not force classes to implement methods they do not need.
   * Example:

     * Keep `LLMClient`, `CodeExecutor`, and `WorkspaceManager` interfaces separate.
     * Avoid “god interfaces”.

5. **Dependency Inversion Principle (DIP)**

   * High-level modules must not depend on low-level modules.
   * Both must depend on abstractions.
   * Example:

     * Agents depend on `LLMClient`, not on `OpenAIClient`.
     * Orchestrator depends on `CodeExecutor`, not on `DockerExecutor`.

### 7.3 Testing Requirements (MANDATORY)

The project must be **fully tested**.

* **Unit tests**

  * Every core component must have unit tests:

    * Orchestrator
    * Agents
    * WorkspaceManager
    * Executors
    * LLM clients (mocked)
  * Tests must not depend on real network access or real LLM APIs.
  * Use mocks or fake implementations where necessary.

* **Test quality**

  * Tests must verify behavior, not implementation details.
  * Each test must:

    * Have a clear purpose
    * Use descriptive test names
    * Avoid duplication
  * No skipped tests or placeholder assertions.

* **Coverage expectations**

  * Core logic must have **high coverage** (≈80%+).
  * Trivial glue code may have lower coverage if justified.

### 7.4 Error Handling and Robustness

* Errors must be:

  * Explicit
  * Informative
  * Typed where appropriate (custom exceptions).
* Do **not** swallow exceptions silently.
* Failures in one agent must not crash the entire system unless explicitly unrecoverable.
* Execution errors (build/test failures) must be captured and reported via structured results.

### 7.5 Architectural Cleanliness

* Enforce **clear layering**:

  * CLI → Orchestrator → Agents → Services (LLM, Workspace, Execution, Memory)
* No circular dependencies between modules.
* No hidden global state (except configuration loaded at startup).
* Use dependency injection explicitly rather than importing concrete implementations directly.

### 7.6 Python Best Practices

* Follow modern Python best practices:

  * Type hints everywhere (`from __future__ import annotations` recommended).
  * `dataclasses` or `pydantic` for data models.
  * No mutable default arguments.
  * No wildcard imports.
* Code must be compatible with:

  * `mypy` or `pyright`
  * `ruff` or `flake8`

### 7.7 Non-Negotiable Constraints for Code Generation

When implementing this project:

* **Do not generate “quick hacks” or shortcuts.**
* **Do not inline large logic blocks inside CLI commands.**
* **Do not mix concerns to reduce file count.**
* **Do not optimize for speed of implementation at the cost of design quality.**

If a design trade-off exists:

* Prefer **clarity, correctness, and extensibility** over minimal code.

Below are **two additional sections** you can append verbatim to `PROJECT_SPEC.md`.
They are written to further constrain Codex toward **high engineering quality**, **predictable progress**, and **low ambiguity**.

Nothing in earlier sections needs to be changed.

---

## 8. Definition of Done (DoD) – PER PHASE AND GLOBAL

This section defines **objective completion criteria**. A phase or the overall project is **not considered complete** unless **all applicable items are satisfied**.

### 8.1 Global Definition of Done (Applies Always)

The entire project is considered **DONE** only if:

1. **Code Quality**

   * All code follows the requirements in Section 7.
   * No TODOs, FIXMEs, or placeholder implementations remain.
   * No commented-out code is present.

2. **Architecture**

   * SOLID principles are demonstrably followed.
   * No circular dependencies between modules.
   * All dependencies point from high-level code to abstractions, never to concrete implementations.

3. **Tests**

   * All tests pass locally via a single command (e.g. `pytest`).
   * No tests rely on:

     * Network access
     * Real LLM APIs
     * Real Docker images (mock where required).
   * Core logic has high coverage (~80%+).

4. **Static Analysis**

   * Code passes:

     * Type checking (`mypy` or `pyright`)
     * Linting (`ruff` or `flake8`)
   * No ignored errors unless explicitly justified and documented.

5. **Usability**

   * `pip install .` works from a clean environment.
   * CLI entry point works as documented.
   * README contains at least one end-to-end usage example.

6. **Safety**

   * File access is sandboxed to the workspace root.
   * Code execution is controlled and explicit.
   * No unsafe shell invocation patterns (e.g. `shell=True`).

---

### 8.2 Phase-Level Definition of Done

Each phase listed in **Section 5 (Implementation Phases)** is considered complete only if:

#### Phase 1 – Project Bootstrap

* Project installs cleanly.
* Empty but valid module structure exists.
* CI-style test run executes (even if minimal tests).

#### Phase 2 – Core Abstractions

* Orchestrator routes messages correctly.
* At least one fake agent can send/receive messages.
* Tests cover message routing and orchestration flow.

#### Phase 3 – Workspace Manager

* Path traversal protection is tested.
* File read/write behavior is deterministic.
* No direct filesystem access outside WorkspaceManager.

#### Phase 4 – Execution Engine

* LocalExecutor tested with mocked subprocess calls.
* DockerExecutor command construction tested.
* ExecutionResult always populated consistently.

#### Phase 5 – LLM Layer

* LLMClient fully abstracted.
* OpenAIClient tested via mocked HTTP.
* No agent imports concrete LLM implementations.

#### Phase 6 – Agents

* Each agent:

  * Has a clear role.
  * Has its own test suite.
  * Can operate using fake/mocked services.
* No agent directly accesses the filesystem or subprocess.

#### Phase 7 – CLI

* CLI commands invoke orchestrator only.
* CLI contains no business logic.
* Invalid user input handled gracefully.

#### Phase 8 – Polishing

* Logging is consistent and structured.
* Error messages are actionable.
* Documentation is complete and accurate.

---

## 9. Reference Agent Workflow (AUTHORITATIVE)

This section defines a **canonical end-to-end workflow**.
Codex must implement behavior that can reproduce this flow without special casing.

### 9.1 High-Level Workflow Overview

```
User
 └── CLI
      └── Orchestrator
           ├── PlannerAgent
           │     └── creates plan
           ├── CodingAgent
           │     └── applies code changes
           ├── TesterAgent
           │     └── runs tests/build
           └── ReviewerAgent
                 └── reviews and approves or rejects
```

The orchestrator is the **only component** allowed to coordinate this flow.

---

### 9.2 Step-by-Step Execution Flow

#### Step 1 – User Task Submission

* User runs:

  ```bash
  multiagent-dev run-task "Add feature X"
  ```
* CLI:

  * Loads configuration.
  * Creates `UserTask`.
  * Instantiates Orchestrator and agents.
  * Sends initial message to `PlannerAgent`.

#### Step 2 – Planning

* PlannerAgent:

  * Receives user task.
  * Queries LLM with:

    * Role description
    * Task description
    * High-level repo context (if available).
  * Produces:

    * Ordered list of steps.
    * Optional acceptance criteria.
  * Stores plan in MemoryService.
  * Sends explicit messages to other agents:

    * CodingAgent: “Implement step 1”
    * TesterAgent: “Prepare to run tests”
    * ReviewerAgent: “Review after implementation”

#### Step 3 – Implementation

* CodingAgent:

  * Reads relevant files via WorkspaceManager.
  * Requests code changes from LLM.
  * Applies changes strictly via WorkspaceManager.
  * Sends completion message to Orchestrator.

#### Step 4 – Testing

* TesterAgent:

  * Runs configured build/test commands via CodeExecutor.
  * Collects ExecutionResult.
  * Summarizes results (pass/fail + key logs).
  * Sends results to Orchestrator and ReviewerAgent.

#### Step 5 – Review

* ReviewerAgent:

  * Inspects modified files.
  * Reviews test results.
  * Uses LLM to produce:

    * Approval OR
    * Change requests.
  * Sends structured decision:

    * `approved: true | false`
    * `comments: [...]`

#### Step 6 – Iteration or Completion

* If rejected:

  * Orchestrator sends feedback to CodingAgent.
  * Flow returns to Step 3.
* If approved:

  * Orchestrator marks task as complete.
  * Final TaskResult returned to CLI.

---

### 9.3 Termination Rules

The workflow must terminate if:

* ReviewerAgent approves changes.
* Or a configurable maximum number of iterations is reached.
* Or a fatal execution error occurs (e.g. infrastructure failure).

Termination must be explicit and logged.

---

### 9.4 Invariants (Must Always Hold)

* Agents **never call each other directly**.
* Agents **never instantiate concrete dependencies**.
* All side effects go through:

  * WorkspaceManager
  * CodeExecutor
  * MemoryService
* Orchestrator is the single source of truth for state.

---

## 10. Final Instruction to Code Generation Models

When implementing this repository:

> **If a design choice exists between “simpler to code” and “cleaner architecture”, always choose the cleaner architecture.**

Correctness, clarity, extensibility, and maintainability are **more important than speed or brevity**.



