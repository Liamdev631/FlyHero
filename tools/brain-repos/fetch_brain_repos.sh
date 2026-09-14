#!/usr/bin/env bash
# Clone every repo listed in the "Brain models and embodied simulation" section of
# https://github.com/cobanov/awesome-fly  (README.md fetched 2026-09-14).
#
# Clones are NOT vendored into this repository (they are ~2.4 GB and most are
# third-party forks of unrelated code). This script plus repo-metadata.tsv is the
# reproducible record; see docs/brain-repo-evaluation.md for the verdicts.
#
# Usage:  bash tools/brain-repos/fetch_brain_repos.sh [destination]
# Default destination: $HOME/projects/awesome-fly-repos
set -u

DEST="${1:-$HOME/projects/awesome-fly-repos}"
mkdir -p "$DEST" || exit 1
cd "$DEST" || exit 1

# slug -> local directory name.  Two repos are both named "fly-brain", so one needs
# a disambiguating local name (eonsystemspbc/fly-brain is the upstream model).
declare -A REPOS=(
  [Drosophila_brain_model]="philshiu/Drosophila_brain_model"
  [fly-brain]="eonsystemspbc/fly-brain"
  [flybody]="TuragaLab/flybody"
  [flygym]="NeLy-EPFL/flygym"
  [flyvis]="TuragaLab/flyvis"
  [webgpu-fly]="abgnydn/webgpu-fly"
  [NeuroFly]="seven-monarchs/NeuroFly"
  [fruit-fly-lab]="vaibhavkedarisetti/fruit-fly-lab"
  [chimera]="caparison1234/chimera"
  [Connectome-OS]="ruvnet/Connectome-OS"
  [FastFly]="eonfathom/FastFly"
  [mps-malecns-model]="seohyunjun/mps-malecns-model"
  [axonweave]="dhakalnirajan/axonweave"
  [fly-brain-erojasoficial]="erojasoficial-byte/fly-brain"
)

fail=0
for name in "${!REPOS[@]}"; do
  slug="${REPOS[$name]}"
  if [ -d "$name/.git" ]; then
    echo "SKIP  $name (already present)"
    continue
  fi
  start=$(date +%s)
  if git clone --depth 1 --single-branch "https://github.com/$slug.git" "$name" >/dev/null 2>&1; then
    echo "OK    $name  ($(( $(date +%s) - start ))s)  <- $slug"
  else
    echo "FAIL  $name  <- $slug"
    fail=$((fail + 1))
  fi
done

echo
echo "cloned into: $DEST"
[ "$fail" -eq 0 ] && echo "all repos present" || echo "$fail repo(s) failed"
exit "$fail"
