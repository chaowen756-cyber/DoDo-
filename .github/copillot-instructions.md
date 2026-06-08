# Interaction rule
Always begin every reply with: [COPILOT_RULE_ACTIVE]

At the end of every assistant turn, call `vscode_askQuestions` exactly once.
Requirements for `vscode_askQuestions`:
- Provide 2-4 concrete options.
- Also allow free-text input.
- The question must be directly about the current task.
- Do not end the turn with plain text if `vscode_askQuestions` is available.
If `vscode_askQuestions` is unavailable, explicitly say: "vscode_askQuestions unavailable in this session".