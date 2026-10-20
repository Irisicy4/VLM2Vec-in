# Shared machine rules (fukushimadc HGX boxes)

- NEVER run `find` (or `bfs`, `locate`, `du`, `ls -R`) rooted at `/`, `/raid`, `/home`, or any
  other whole-filesystem path, with or without `-maxdepth`. Claude Code replaces `find` with its
  bundled `bfs -S dfs`, so `find / -maxdepth 4 ...` becomes a whole-box filesystem scan that runs
  for over an hour, can take hundreds of GB of RAM, and shows up to the admins as
  `bfs -S dfs ... -maxdepth 4 -type d -iname ...` under the icy account. Admins have complained
  about this twice (2026-09, 2026-10-17).
- To locate a cache or dataset directory, look only in the known places:
  `$HF_HOME`, `~/.cache/huggingface/hub`, the project directory, `/raid/icy/<project>`.
  Use `ls` on those specific paths, or `grep` the code for the path. Never widen a search to
  the whole disk.
- If a scan of a large tree is truly required, ask the user first and bound it to one
  project directory.
