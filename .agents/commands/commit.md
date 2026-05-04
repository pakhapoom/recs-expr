---
name: commit
description: Analyze uncommitted changes and produce a logical commit plan without modifying git state.
allowed-tools: Bash(git status:*), Bash(git diff:*), Read
---

# Instructions

Analyze all uncommitted changes and untracked files, and produce a commit plan. Run only inspection commands; do not run `git add`, `git commit`, or any other command that modifies git state.

## Steps

1. Run `git status --short` to list all changed and untracked files.
2. Run `git diff HEAD` to read the full staged and unstaged diff.
3. For untracked files that matter, run `git diff --no-index /dev/null <file>` or read them directly.
4. Group files into logical commits. Files that change together for one reason belong in one commit. Unrelated changes must be in separate commits. Apply judgement, and err toward fewer coherent commits over many tiny ones.
5. For each commit, pick a message following the Conventional Commits spec:
   - Format: `<type>(<optional scope>): <short description>`
   - Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `style`, `build`, `ci`, `perf`
   - Description: imperative mood, lowercase, no period, max ~72 characters total for the first line
   - No body, no footer; one line only
6. Output the plan as markdown, one section per commit, numbered sequentially:

```markdown
# commit 1
git add <file1> <file2>
git commit -m "<message>"

# commit 2
git add <file3>
git commit -m "<message>"
```

Do not include explanatory prose outside the code blocks. Do not run `git add` or `git commit`.
