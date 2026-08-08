#!/bin/sh
set -e

YEAR=2026
DIR=/tmp/install-tl

apt-get update
apt-get install -y wget ca-certificates perl

mkdir -p "$DIR"
cd "$DIR"

wget -qO- https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
  | tar xzf - --strip-components=1

cat > texlive.profile <<EOF
selected_scheme scheme-full
TEXDIR /usr/local/texlive/$YEAR
option_doc 0
option_src 0
EOF

./install-tl -profile texlive.profile
/usr/local/texlive/$YEAR/bin/x86_64-linux/tlmgr path add

rm -rf "$DIR"
