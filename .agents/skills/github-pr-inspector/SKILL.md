---
name: github-pr-inspector
description: >-
  Inspect, audit, and manage GitHub pull requests, CI workflow statuses, and git remote integration in Antigravity.
  Use when the user or agent needs to verify GitHub connectivity, check open/merged PRs, monitor GitHub Actions CI runs,
  or create and merge pull requests using the gh CLI.
---

# GitHub PR Inspector & Lifecycle Skill

This skill provides procedures and runbooks for inspecting, creating, and verifying pull requests and GitHub integration directly within Antigravity.

---

## 1. Verify GitHub Connection
Check authentication status and authorized scopes:
```bash
gh auth status
```
*Expected: `✓ Logged in to github.com account <username>` with `repo` and `workflow` scopes.*

---

## 2. Inspect Pull Requests
List all PRs (open, merged, closed):
```bash
gh pr list --state all --limit 20
```

Inspect details and review comments of a specific PR:
```bash
gh pr view <pr_number>
```

View git diff associated with a PR:
```bash
gh pr diff <pr_number>
```

---

## 3. Monitor GitHub Actions CI Status
Check live CI status on any open PR:
```bash
gh pr checks <pr_number>
```
*Wait for all required checks (e.g. `FastAPI Telemetry`, `Rust Edge Vision`) to report `✓ successful` before merging.*

---

## 4. PR Creation & Clean Merge Workflow
When completing a feature branch:
1. Push branch to origin:
   ```bash
   git push -u origin feat/<task-name>
   ```
2. Create Pull Request:
   ```bash
   gh pr create --base main --head feat/<task-name> --title "feat(<scope>): <summary>" --body "### Summary ... ### Verification ..."
   ```
3. Await CI pass:
   ```bash
   gh pr checks <pr_number>
   ```
4. Merge cleanly into `main`:
   ```bash
   gh pr merge <pr_number> --merge
   ```
5. Synchronize local `main`:
   ```bash
   git checkout main && git pull origin main
   ```
