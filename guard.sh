#!/bin/sh
while true; do
  clear;
  uv run clj -M:test "$@";
  git ls-files | xargs inotifywait -e close_write;
done
