# Purging the committed AI conversation logs from git history

## What is in there

Commit `b86b285` added 22 JSON exports of assistant runs under
`scratch/conversations-backup-20260830-042951/`. Each holds the prompt, the
model's reply, run and conversation UUIDs, model names, timings, and in at least
one case an operator's name (`S. Kulkarni`).

The prompts are synthetic demo material — a fabricated P&ID `A-101-001 Rev 6`,
not real MRPL drawings — so this is a professionalism and privacy problem rather
than a disclosure of confidential plant data. Treat it as the former.

## What has already been done

- The files are **untracked** and will not be committed again.
- `.gitignore` blocks `scratch/conversations-backup-*/` and `scratch/*.db`.
- The 22 files are still on disk locally, and still reachable in history at
  `b86b285`.

## What is left, and why it was not done automatically

Removing them from history rewrites every commit from `b86b285` onward and
requires a **force-push**. This repository has two remotes:

```
origin     https://github.com/Straw-hat-Luffy26/Arjun
shreyash   https://github.com/ShreyashPatil123/Arjun
```

so a rewrite invalidates every clone held by either owner. That is a
coordination decision, not a code change, and it needs a clean working tree —
which this one has not had while the SIH fixes were in progress.

## The procedure

**1. Land or stash everything first.** `git status` must be clean.

**2. Tell the other remote owner.** After the force-push, anyone with a clone
re-clones, or runs `git fetch origin && git reset --hard origin/main`. A plain
`git pull` will produce a merge that puts the logs straight back.

**3. Back up the current history**, so the rewrite is reversible:

```bash
git branch backup/pre-log-purge
```

**4. Rewrite.** `git filter-repo` is the maintained tool and is not installed
here; install it (`pip install git-filter-repo`) and prefer it:

```bash
git filter-repo --path scratch/conversations-backup-20260830-042951 --invert-paths --force
```

Without it, `git filter-branch` still ships with git and does the same job:

```bash
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch scratch/conversations-backup-20260830-042951" --prune-empty --tag-name-filter cat -- --all
```

**5. Verify nothing is reachable.** This must print nothing:

```bash
git log --all --oneline -- scratch/conversations-backup-20260830-042951
```

**6. Expire the reflog and repack**, or the objects stay in the local repo:

```bash
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

**7. Force-push both remotes.**

```bash
git push origin --force --all
```

```bash
git push shreyash --force --all
```

**8. Delete the backup branch** once both remotes and every clone are confirmed
good — and not before:

```bash
git branch -D backup/pre-log-purge
```

## If the rewrite goes wrong

Before step 8, everything is recoverable:

```bash
git reset --hard backup/pre-log-purge
```

## Worth knowing

GitHub keeps rewritten commits reachable by SHA on its side for a while, and
forks are not rewritten at all. If the content genuinely mattered, a rewrite
alone would not be sufficient — you would open a GitHub support request to purge
the cached views. For synthetic demo prompts that is not warranted; for anything
real it would be.
