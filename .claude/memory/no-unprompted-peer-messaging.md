---
name: no-unprompted-peer-messaging
description: Do not message the paper/writing session (or other peer sessions) unless the user explicitly asks; unprompted cross-session traffic is noise.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4432a7aa-e1e1-4d8e-8bb0-e803fd5ea440
  modified: 2026-09-12T16:57:14.419Z
---

Do not send messages to the paper-writing session without the user explicitly asking. On 2026-09-12
the user said: "dont talk to paper session wihout my ask from now, you brought too much noise."

**Why:** over one session I sent roughly a dozen unprompted updates to the paper session and a similar
number to the text-side collaborator, including corrections of my own corrections. Each one demanded
the recipient's attention and most were not requested. The peer had also repeatedly written "nothing
needed from you" while I kept replying. Volume displaced signal, and the user is the one who has to
live with the resulting churn in the shared documents.

**How to apply:** default to writing findings into the repo (`mmrag/RL_REDESIGN.md` ledger,
`mmrag/paper.md`) and reporting to the user, not to peers. Message a peer only when the user asks, or
when a peer asks me a direct question. Treat "nothing needed from you" as a stop signal, not an
opening. If something genuinely must reach a peer — a retraction of a number they are already
publishing — batch it and say so in one message rather than a stream. Same restraint applies to the
text-side collaborator even though the user named only the paper session.

Related: [[ledger-everything-including-nulls]]
