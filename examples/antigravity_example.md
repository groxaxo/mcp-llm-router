# Antigravity Agent Example Workflows

Real-world examples of using the MCP LLM Router within Antigravity agents.

## Example 1: Multi-Model Code Review Agent

```python
# Start a code review session
session = start_session(
    goal="Comprehensive code review of the authentication module",
    constraints="Check security, performance, and code quality",
    metadata={"module": "auth", "files": ["auth.py", "jwt_utils.py"]}
)
sid = session["session_id"]

# Step 1: Use a fast model for initial scan
log_event(sid, "info", "Starting initial code analysis")
quick_scan = agent_llm_request(
    session_id=sid,
    prompt=f"Review this authentication code:\n{auth_code}\nList potential issues.",
    model="gpt-3.5-turbo",
    provider="openai",
    api_key_env="OPENAI_API_KEY",
    temperature=0.3
)

# Step 2: Use specialized model for security analysis
log_event(sid, "info", "Deep security analysis in progress")
security_review = agent_llm_request(
    session_id=sid,
    prompt=f"Perform security audit:\n{auth_code}\nFocus on vulnerabilities.",
    model="anthropic/claude-3-opus-20240229",
    provider="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    system_prompt="You are a security expert specializing in authentication systems.",
    temperature=0.2
)

# Step 3: Generate comprehensive report with GPT-4
log_event(sid, "info", "Generating final report")
final_report = agent_llm_request(
    session_id=sid,
    prompt=f"Combine these reviews into a comprehensive report:\n\nInitial: {quick_scan['content']}\n\nSecurity: {security_review['content']}",
    model="gpt-4-turbo",
    provider="openai",
    api_key_env="OPENAI_API_KEY",
    temperature=0.7,
    max_tokens=3000
)

# Log completion
log_event(
    sid, 
    "success", 
    "Code review complete", 
    details={
        "models_used": ["gpt-3.5-turbo", "claude-3-opus", "gpt-4-turbo"],
        "total_findings": extract_findings_count(final_report['content'])
    }
)
```

## Example 2: Incremental Feature Development

```python
# Start development session
session = start_session(
    goal="Add email verification to user registration",
    constraints="Use existing SMTP config, add database migration, write tests"
)
sid = session["session_id"]

# Phase 1: Architecture planning
log_event(sid, "info", "Planning architecture")
architecture = agent_llm_request(
    session_id=sid,
    prompt="Design email verification flow for user registration with database schema changes",
    model="gpt-4",
    provider="openai",
    api_key_env="OPENAI_API_KEY",
    temperature=0.4
)

# Phase 2: Generate migration with specialized code model
log_event(sid, "info", "Generating database migration")
migration = agent_llm_request(
    session_id=sid,
    prompt=f"Generate SQLAlchemy migration for: {architecture['content']}",
    model="deepseek/deepseek-coder-33b-instruct",
    provider="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    temperature=0.1  # Low temp for code generation
)

# Write migration file
# ... file operations ...

log_event(sid, "success", "Migration generated", details={"file": "migrations/add_email_verification.py"})

# Phase 3: Generate implementation
log_event(sid, "info", "Implementing email service")
implementation = agent_llm_request(
    session_id=sid,
    prompt=f"Implement email verification service based on:\n{architecture['content']}",
    model="gpt-4",
    provider="openai",
    api_key_env="OPENAI_API_KEY",
    temperature=0.2
)

# Phase 4: Generate tests
log_event(sid, "info", "Writing tests")
tests = agent_llm_request(
    session_id=sid,
    prompt=f"Write pytest tests for:\n{implementation['content']}",
    model="anthropic/claude-3-sonnet-20240229",
    provider="openrouter",
    api_key_env="OPENROUTER_API_KEY"
)

# Review session context
context = get_session_context(sid)
print(f"Completed {len(context['session']['events'])} steps")
```

## Example 3: Error Recovery Loop

```python
session = start_session(
    goal="Fix failing integration tests",
    constraints="Minimal code changes, preserve existing behavior"
)
sid = session["session_id"]

max_attempts = 3
attempt = 0

while attempt < max_attempts:
    attempt += 1
    log_event(sid, "info", f"Fix attempt {attempt}/{max_attempts}")
    
    # Get test output (simulated)
    test_output = run_tests()  # Your test runner
    
    if test_output["success"]:
        log_event(sid, "success", "All tests passing")
        break
    
    # Log the failure
    log_event(
        sid,
        "error",
        f"Tests failed: {test_output['failed_count']} failures",
        details={"output": test_output["error_log"]}
    )
    
    # Ask LLM for fix suggestions
    fix_suggestion = agent_llm_request(
        session_id=sid,
        prompt=f"""Test failures:
{test_output['error_log']}

Relevant code:
{test_output['failing_test_code']}

Suggest minimal fixes to make tests pass.""",
        model="gpt-4",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        temperature=0.3
    )
    
    # Apply fix (implement your own apply_fix logic)
    apply_fix(fix_suggestion['content'])
    log_event(sid, "info", "Applied suggested fix")

# Final report
context = get_session_context(sid)
if test_output["success"]:
    print(f"✓ Tests fixed in {attempt} attempts")
else:
    print(f"✗ Tests still failing after {attempt} attempts")
    print("Review session context for debugging:")
    print(json.dumps(context, indent=2))
```

## Example 4: Cross-Provider Model Selection

```python
session = start_session(
    goal="Generate documentation for API endpoints",
    constraints="OpenAPI spec format, include examples"
)
sid = session["session_id"]

# Define model preferences based on task
models = [
    {
        "name": "gpt-4",
        "provider": "openai",
        "api_key": "OPENAI_API_KEY",
        "use_for": "complex reasoning"
    },
    {
        "name": "anthropic/claude-3-haiku-20240307",
        "provider": "openrouter",
        "api_key": "OPENROUTER_API_KEY",
        "use_for": "quick drafts"
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "provider": "deepinfra",
        "api_key": "DEEPINFRA_API_KEY",
        "use_for": "cost-effective generation"
    }
]

# Try DeepInfra first (cheapest)
log_event(sid, "info", "Attempting with Llama 3.1 (DeepInfra)")
try:
    result = agent_llm_request(
        session_id=sid,
        prompt="Generate OpenAPI documentation for these endpoints:\n" + endpoint_code,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        provider="deepinfra",
        api_key_env="DEEPINFRA_API_KEY",
        max_tokens=4000
    )
    
    if result["success"]:
        log_event(sid, "success", "Documentation generated with Llama 3.1")
    else:
        raise Exception("DeepInfra failed")
        
except Exception as e:
    # Fallback to Claude Haiku
    log_event(sid, "warning", "DeepInfra failed, trying Claude Haiku")
    result = agent_llm_request(
        session_id=sid,
        prompt="Generate OpenAPI documentation for these endpoints:\n" + endpoint_code,
        model="anthropic/claude-3-haiku-20240307",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4000
    )
    
    if not result["success"]:
        # Final fallback to GPT-4
        log_event(sid, "warning", "Claude failed, using GPT-4")
        result = agent_llm_request(
            session_id=sid,
            prompt="Generate OpenAPI documentation for these endpoints:\n" + endpoint_code,
            model="gpt-4",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4000
        )
        log_event(sid, "success", "Documentation generated with GPT-4")
```

## Example 5: Session Context Analysis

```python
# Long-running development session
session = start_session(
    goal="Refactor monolithic app to microservices",
    constraints="Maintain API compatibility, zero-downtime migration"
)
sid = session["session_id"]

# ... Many operations happen ...
# File changes, tests, deployments, etc.

# Periodically analyze session context for insights
context = get_session_context(sid)
session_data = context["session"]

# Count events by type
event_counts = {}
for event in session_data["events"]:
    kind = event["kind"]
    event_counts[kind] = event_counts.get(kind, 0) + 1

# Use LLM to analyze the session
analysis = agent_llm_request(
    session_id=sid,
    prompt=f"""Analyze this development session:

Goal: {session_data['goal']}
Duration: {calculate_duration(session_data)}
Events: {json.dumps(event_counts)}
Recent events: {json.dumps(session_data['events'][-10:], indent=2)}

Provide:
1. Progress assessment
2. Blockers or risks
3. Recommended next steps
""",
    model="gpt-4",
    provider="openai",
    api_key_env="OPENAI_API_KEY",
    temperature=0.5
)

print("Session Analysis:")
print(analysis["content"])
```

## Tips for Antigravity Integration

### 1. **Provider Selection Strategy**

- **OpenAI GPT-4**: Complex reasoning, architecture decisions, final reviews
- **OpenAI GPT-3.5**: Quick lookups, simple transformations, drafts
- **Anthropic Claude (via OpenRouter)**: Long context tasks, nuanced analysis
- **DeepSeek Coder (via OpenRouter)**: Code generation, refactoring
- **Llama 3.1 (via DeepInfra)**: Cost-effective bulk operations

### 2. **Temperature Guidelines**

- `0.1-0.2`: Code generation, migrations, config files
- `0.3-0.5`: Bug fixes, documentation, refactoring
- `0.6-0.8`: Creative tasks, brainstorming, exploration
- `0.9-1.0`: Diverse test data generation

### 3. **Session Organization**

Create separate sessions for:
- Feature development (one session per feature)
- Bug fixes (one session per bug)
- Refactoring (one session per module)
- Code reviews (one session per review)

### 4. **Logging Best Practices**

```python
# Good: Detailed, structured
log_event(
    sid, 
    "error", 
    "Database migration failed",
    details={
        "migration": "0042_add_indexes.py",
        "error": str(e),
        "affected_tables": ["users", "sessions"]
    }
)

# Bad: Vague
log_event(sid, "error", "Something went wrong")
```

### 5. **Cost Optimization**

```python
# Cheap model for first pass
draft = agent_llm_request(sid, prompt, model="gpt-3.5-turbo", provider="openai")

# Expensive model only if needed
if quality_check(draft) < threshold:
    final = agent_llm_request(sid, prompt, model="gpt-4", provider="openai")
```
