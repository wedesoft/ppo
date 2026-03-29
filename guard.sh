#!/bin/sh
while true; do
  clear;
  xvfb-run clj -M:test "$@";
  git ls-files | xargs inotifywait -e close_write;
done
