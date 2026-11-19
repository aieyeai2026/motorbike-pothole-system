# Antigravity Agent Guidelines & Git/PR Integration

## 1. Project Context & Workspace Boundary
- This repository is `motorbike-pothole-system`.
- **Strict Sandbox Boundary**: All file inspection, edits, and terminal commands must operate strictly inside the repository workspace root (`/Users/gautam/motorbike-pothole-system`). Never attempt to read, write, or inspect external system or application directories (such as `~/.gemini/antigravity/brain/`) to avoid triggering unnecessary permission dialogues.

## 2. Strict 8-Stage Gated Engineering Pipeline
Every engineering task must follow this exact 8-stage gate:
1. **Branch & Isolate**: Create a distinct feature branch off `main` named `feat/<task-name>` (or `fix/<bug-name>`).
2. **Contract & Interface First**: Define schemas, data models, and API function signatures.
3. **Red Test Suite**: Write unit and integration test assertions verifying expected behavior, edge cases, and failure modes before implementing logic.
4. **Implement Minimal Code**: Write clean, modular, production-ready code satisfying the test criteria.
5. **Green Verification**: Execute automated test suites locally (`cargo test`, `flutter test`, `pytest`) and ensure 100% pass (0 failures).
6. **Lint & Static Analysis**: Verify zero lint warnings, formatting errors, or compiler warnings.
7. **Document & Status**: Record commit hashes, test results, and status in the milestone status document (`STATUS-PHASE-X.md`).
8. **PR & Clean Merge to Main**:
   - Push the feature branch to `origin`: `git push -u origin feat/<task-name>`.
   - Create a Pull Request using GitHub CLI (`gh pr create --title ... --body ...`).
   - Merge the PR cleanly into `main` (`gh pr merge --merge` or fast-forward/clean merge) and push `main` to `origin`.

## 3. Git & GitHub CLI Integration Flow
- **CLI Tool**: Use GitHub CLI (`gh`) for PR creation, review, and merging.
- **Commit Messages**: Follow Conventional Commits format:
  - `feat(<subsystem>): <imperative summary> (Task X.Y, T-X.Y.Z)`
  - `test(<subsystem>): <test additions>`
  - `fix(<subsystem>): <bugfix summary>`
- **PR Description Template**:
  - Summary of changes and components touched.
  - Linked test IDs and verification output (e.g. `pytest` result quote).
  - Milestone status report reference.
