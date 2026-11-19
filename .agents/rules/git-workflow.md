---
description: Enforce strict git branching, PR lifecycle, and test-driven workflow in Antigravity
trigger: always_on
---

# Git & PR Lifecycle Workflow in Antigravity

Whenever implementing a new feature or task in this repository:
1. Always verify current branch with `git branch --show-current`.
2. Work exclusively on task feature branches (`feat/<task-name>`).
3. Write test cases first (TDD) corresponding to the test plan (`TEST-PLAN.md`).
4. Implement minimal clean code to make tests pass.
5. Verify tests with 100% pass rate locally before pushing.
6. Push feature branch to `origin` using `git push -u origin <branch-name>`.
7. Create a GitHub PR using `gh pr create`.
8. Merge the PR into `main` after verification and push `main` to `origin`.
