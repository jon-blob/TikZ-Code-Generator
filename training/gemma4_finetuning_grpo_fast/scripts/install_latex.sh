#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/latex"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if compgen -G "$DEST/bin/*/pdflatex" >/dev/null; then
  echo "TeX Live is already installed in $DEST"
  exit 0
fi

curl -L https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
  | tar -xz -C "$TMP" --strip-components=1

cat > "$TMP/texlive.profile" <<EOF
selected_scheme scheme-full
TEXDIR $DEST
TEXMFCONFIG $DEST/texmf-config
TEXMFHOME $DEST/texmf-home
TEXMFLOCAL $DEST/texmf-local
TEXMFSYSCONFIG $DEST/texmf-config
TEXMFSYSVAR $DEST/texmf-var
TEXMFVAR $DEST/texmf-var
option_doc 0
option_src 0
EOF

perl "$TMP/install-tl" -profile "$TMP/texlive.profile"
echo "Full TeX Live installed in $DEST"
