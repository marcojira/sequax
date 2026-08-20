# AGENTS.md

Project context/state lives in notes/CONTEXT.md, you must read it when resuming work. If it doesn't exist, it is a new project.

Use plain language, short sentences, and avoid dense or overly compressed phrasing.

# WORKING PRINCIPLES

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- **If the request is complicated, break it down with me and ask questions. DO NOT THINK FOR TOO LONG.**

## 2. One Task, One Small Change (surgical)

**Touch only what you must. Clean up only your own mess.**

- **ONLY DO ONE TASK PER REQUEST.** If there are additional tasks you think are important/related, mention them but don't execute them.
- **Make one small change at a time.** Making more is overwhelming and will lead to the change being rejected.
- Minimum code that solves the problem. Nothing speculative.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Simplicity First

PREFER SIMPLICITY OVER ALL ELSE. THINK OF SIMPLER WAYS TO WRITE THE SAME CODE.
CODE MUST BE EASY TO UNDERSTAND.

- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

# CODE STYLE

- Group similar sections of code together within functions

## Comments & Documentation
- **Avoid excessive comments** - code should be self-documenting with clear variable/function names
- Only add comments:
  - The code is complex or non-obvious
  - The code is not self-explanatory despite clear naming
  - There are long sections of code where adding comments to separate each section is clearer
- **Avoid excessive docstrings** - only document non-obvious functions
- Write a short (usually single line summary) docstring at the top of each function
