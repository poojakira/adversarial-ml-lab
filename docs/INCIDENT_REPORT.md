# Incident Report — Workspace Directory Wipe During Initial Build

- **Report ID:** INC-2026-07-11-01
- **Date of incident:** 2026-07-11
- **Author:** poojakira
- **Severity:** SEV-2 (work-in-progress destroyed; fully recovered, no data shipped/lost permanently)
- **Status:** Resolved
- **Component:** Repository bootstrap / source-control tooling (sandbox clone tooling)

---

## 1. Summary

While bootstrapping `adversarial-ml-lab` for the first time, the entire working
directory `/projects/sandbox/adversarial-ml-lab` — source, tests, the local git
history (8 commits), and the Python virtualenv — was deleted mid-session. The
deletion was a side effect of invoking a repository setup/clone operation that
targets a **fixed** destination path: it cleared the existing directory contents
in preparation for a clone, and then the clone itself failed because the remote
GitHub repository did not yet exist. The net result was an empty directory and a
loss of all uncommitted-to-remote work.

All work was reconstructed from source content held in the session and
re-verified (13/13 tests passing, benchmark reproduced bit-for-bit). No
production system, no remote branch, and no third-party data were affected.

---

## 2. Impact

| Dimension | Impact |
|-----------|--------|
| Data | Local-only: all files + local git history for the project were destroyed. Nothing had been pushed to a remote yet, so no remote data was lost. |
| Availability | None — no deployed service exists yet. |
| Users | None external. Single-developer build session only. |
| Recovery cost | ~1 rebuild cycle: recreate 20 files, recreate the venv, reinstall CPU PyTorch, re-run the full test + benchmark suite, recreate the 8-commit history. |
| Confidentiality | None — no secrets exposed. Auth tokens are injected by tooling and were never written to disk. |

---

## 3. Timeline (UTC, 2026-07-11)

1. **~02:48** — Project scaffolded: `pyproject.toml`, `src/adv_lab/**`, `tests/**`
   written. Dependencies installed into `.venv` (Python 3.12.13, torch 2.13.0+cpu).
2. **~02:52** — Full suite verified: **13/13 tests pass**; `adv-eval` CLI produces
   `results/report.json` (clean 0.920 / FGSM 0.508 / PGD 0.500 / C&W 0.034).
3. **~02:55** — README/CHANGELOG/CI added; local git initialized; **8 commits**
   created in the intended sequence; working tree clean.
4. **Push attempt** — Push failed: no `origin` remote configured, then
   "Remote URL of the origin not supported" after adding a plain `github.com`
   URL (the tool requires its gateway URL form).
5. **Existence probe** — A repository setup/clone call was made to determine
   whether the remote repo existed. Because the destination directory already
   existed, the tool **removed the directory contents** before attempting the
   clone, then the clone **failed with "repository not found"** (the GitHub repo
   `poojakira/adversarial-ml-lab` did not exist yet).
6. **Detection** — Immediately afterward, shell commands scoped to the project
   directory failed with `bwrap ENOENT` / `No such file or directory`. A
   directory listing confirmed `/projects/sandbox/adversarial-ml-lab` was gone.
7. **Recovery** — All 20 files were rewritten from known-good content; the venv
   was recreated and dependencies reinstalled; tests re-run (**13/13 pass**);
   benchmark reproduced **identically** (deterministic seed); the 8-commit
   history was recreated in the same sequence. Working tree clean.
8. **Root-caused** — Confirmed the remote repo did not exist and the gateway
   does not auto-create repos; the wipe was attributed to the clone tool clearing
   a non-empty fixed target path prior to a failing clone.

---

## 4. Root Cause

**Primary cause:** A clone/`repo_set_up` operation was invoked against a fixed
destination path (`/projects/sandbox/adversarial-ml-lab`) that already contained
active work. The tool's behavior is to prepare that path for a fresh clone by
clearing it first; when the subsequent clone failed (remote repository did not
exist), the directory was left empty with no automatic rollback.

**Contributing factors:**

- **No remote repository existed yet.** The GitHub repo had not been created, so
  any clone was guaranteed to fail — turning a "probe" into a destructive no-win.
- **Using a destructive tool to test for existence.** Repository existence was
  probed with a tool that mutates the filesystem, rather than a read-only check.
- **Work not yet backed up off-machine.** The local commits existed only in the
  sandbox; there was no remote to fall back on.
- **Fixed, shared destination path.** The tool always operates on one path, so it
  could not be redirected to a scratch location to make the probe safe.

---

## 5. Resolution & Recovery

- Rebuilt all source, tests, packaging, docs, and CI from source content.
- Recreated `.venv` and reinstalled `torch`/`torchvision`/`numpy` (CPU wheels).
- Re-ran `pytest` → **13 passed**; re-ran `adv-eval` → identical JSON report,
  confirming the rebuild is faithful (deterministic seeds).
- Recreated the 8-commit history in the original order; verified clean tree.
- Set `origin` to the gateway URL form accepted by the push tooling.

**Current state:** fully recovered and ready to push once the remote repository
exists.

---

## 6. Corrective & Preventive Actions

| # | Action | Type | Owner | Status |
|---|--------|------|-------|--------|
| 1 | Never invoke clone/`repo_set_up` against a non-empty working directory; treat those tools as destructive to their target path. | Preventive | poojakira | Done (practice adopted) |
| 2 | Confirm a remote repository exists **before** any clone/push, using read-only checks — do not use a destructive clone to test existence. | Preventive | poojakira | Done |
| 3 | Push a branch to the remote as early as possible so local commits are backed up off-machine. | Preventive | poojakira | In progress (this push) |
| 4 | Create the empty remote repo first (no README/license) so the initial push is a fast-forward. | Corrective | poojakira | Done (repo created) |
| 5 | Keep the build deterministic (fixed seeds) so any future rebuild is verifiable and reproducible. | Preventive | poojakira | Done |

---

## 7. Lessons Learned

- **Probing with a destructive tool is itself the risk.** Existence should be
  checked with read-only operations; a "check" that can delete state is not a
  check.
- **Off-machine backup is the real safety net.** Local-only commits offer no
  protection against a workspace wipe — push early.
- **Deterministic builds turn a disaster into an inconvenience.** Because the
  benchmark and tests are seeded, the rebuild could be proven identical to the
  original, which is what made recovery trustworthy rather than merely plausible.
