#!/usr/bin/env bash
# Create the private GitHub repo and push, with the authorship checks required by the project rules.
# Prerequisite (once): gh auth login
set -euo pipefail
cd "$(dirname "$0")"
OWNER="${OWNER:-basilll007}"
REPO="${REPO:-nio-cybersec-ml}"

gh auth status >/dev/null 2>&1 || { echo "gh is not logged in: run 'gh auth login' first"; exit 1; }

# 1. no co-author / generated-by / bot identity anywhere in history
if git log --format='%an <%ae>%n%b' | grep -iE 'co-authored|generated with|noreply@anthropic'; then
  echo "ABORT: forbidden attribution found in history"; exit 1
fi
authors=$(git log --format='%an <%ae>' | sort -u)
echo "commit authors: $authors"
[ "$(echo "$authors" | wc -l)" -eq 1 ] || { echo "ABORT: more than one author"; exit 1; }

# 2. never push data / secrets
if git ls-files | grep -iE '(^data/|\.mp4$|^frames/|kaggle\.json|\.env$|\.pt$|\.pth$)'; then
  echo "ABORT: forbidden file tracked"; exit 1
fi

# 3. create (private) and push
if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$OWNER/$REPO" --private --source . --remote origin
fi
git push -u origin "$(git branch --show-current)"

# 4. contributor check: must list only the owner
sleep 5
echo "contributors:"; gh api "repos/$OWNER/$REPO/contributors" --jq '.[].login'
echo "repo: https://github.com/$OWNER/$REPO"
