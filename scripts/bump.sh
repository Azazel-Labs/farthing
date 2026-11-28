#!/usr/bin/env bash
# Raise farthing's version in Cargo.toml, the one place it's set, and the lockfiles with it. Once a commit that changes
# it is pushed to main and passes CI, CI tags that commit vX.Y.Z and releases it.
#
#   scripts/bump.sh patch        # 1.2.3 -> 1.2.4 (1.3.0-rc.1 -> 1.3.0)
#   scripts/bump.sh minor        # 1.2.3 -> 1.3.0
#   scripts/bump.sh major        # 1.2.3 -> 2.0.0
#   scripts/bump.sh 2.0.0-rc.1   # exactly this
#
# Then commit what it changed (git commit -am "Release X.Y.Z") and push.
set -euo pipefail

CRATE=farthing
SECTION='workspace.package'   # the Cargo.toml table that holds the version
LOCKS=(engine-poc/Cargo.toml demo-web/Cargo.toml bench-compare/Cargo.toml)       # other manifests whose lockfiles name the crate

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
die() { echo "bump: $*" >&2; exit 1; }
semver='^([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z.-]+)?$'

[ $# -eq 1 ] || usage
cd "$(git rev-parse --show-toplevel)"
current="$(sed -n "/^\[$SECTION\]/,/^\[/ s/^version = \"\([^\"]*\)\".*/\1/p" Cargo.toml | head -1)"
[[ "$current" =~ $semver ]] || die "Cargo.toml's [$SECTION] version, '$current', isn't X.Y.Z"
major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}" patch="${BASH_REMATCH[3]}" pre="${BASH_REMATCH[4]}"

case "$1" in
  patch) if [ -n "$pre" ]; then next="$major.$minor.$patch"; else next="$major.$minor.$((patch + 1))"; fi ;;
  minor) next="$major.$((minor + 1)).0" ;;
  major) next="$((major + 1)).0.0" ;;
  -*) usage ;;
  *) next="${1#v}"; [[ "$next" =~ $semver ]] || die "'$1' isn't patch, minor, major or a version like 1.2.3" ;;
esac

# Newer: a higher X.Y.Z, or the same one as a release of its pre-release, or a later pre-release of it.
base() { echo "${1%%-*}"; }
newer=false
if [ "$(base "$next")" != "$(base "$current")" ]; then
  [ "$(printf '%s\n%s\n' "$(base "$current")" "$(base "$next")" | sort -V | tail -1)" = "$(base "$next")" ] && newer=true
elif [[ "$current" == *-* && "$next" != *-* ]]; then
  newer=true
elif [[ "$current" == *-* && "$next" == *-* && "$current" != "$next" ]]; then
  [ "$(printf '%s\n%s\n' "$current" "$next" | sort -V | tail -1)" = "$next" ] && newer=true
fi
$newer || die "$next isn't newer than Cargo.toml's $current"
git rev-parse -q --verify "refs/tags/v$next" >/dev/null && die "v$next is already released"

sed -i "/^\[$SECTION\]/,/^\[/ s/^version = \"[^\"]*\"/version = \"$next\"/" Cargo.toml
# A crate published beside another depends on it by path and version, and Cargo can't take that version from the
# workspace, so those requirements move here too.
sed -i -E "/path = /s/version = \"=?${current//./\\.}\"/version = \"$next\"/" $(git ls-files '*Cargo.toml')
# CI and publishing build with --locked, so the lockfiles move with it; only this project's own entries change.
cargo update --workspace --quiet
for manifest in "${LOCKS[@]}"; do
  cargo update --quiet -p "$CRATE" --manifest-path "$manifest"
done

echo "$CRATE is $next (was $current). Commit it (git commit -am \"Release $next\") and push to main:"
echo "once CI passes, that commit is tagged v$next and released."
