#!/usr/bin/env bash
# Deploy the site to GitHub Pages: mirrors web/ onto the gh-pages branch, keeping the
# chunked model weights (web/weights-remote/, built by quant/split_weights.py) alongside it.
# Run from the repo root in Git Bash:  ./deploy.sh
set -e
cd "$(dirname "$0")"

git worktree remove -f .ghp 2>/dev/null || true
git worktree add .ghp gh-pages

# wipe everything except .git and the (huge, rarely-changing) weight parts
find .ghp -mindepth 1 -maxdepth 1 ! -name .git ! -name weights-remote -exec rm -rf {} +
# copy the site, minus dev-only weight copies
tar -C web --exclude=weights-qwen3 --exclude=weights-remote -cf - . | tar -C .ghp -xf -
# refresh the weight parts (no-op commit if unchanged)
mkdir -p .ghp/weights-remote
cp -f web/weights-remote/* .ghp/weights-remote/

cd .ghp
git add -A
git commit -m "deploy site" || echo "site unchanged"
git push origin gh-pages
cd ..
git worktree remove -f .ghp
echo "deployed -> https://blazingphoenix7.github.io/bitcrush/"
