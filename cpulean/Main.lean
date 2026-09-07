import Reap.Agentic

open Reap.Agentic

def main : IO Unit := do
  let evs : Array Evidence := #[
    { id := "p-l0", kind := .linkSearch, weight := 0.30,
      payload := "https://example.org/lemma:sq_connected", sourceDesc := "opencode:link" },
    { id := "p-s0", kind := .leanSearch, weight := 0.25,
      payload := "lemma sq_connected", sourceDesc := "opencode:lean-search" },
    { id := "p-t0", kind := .toolCall, weight := 0.20,
      payload := "lake env lean smoke.lean ok", sourceDesc := "opencode:tool" }
  ]
  let rw := reweightBayes evs
  IO.println s!"[ring] phi=tanh(sum w) = {rw.phi}"
  IO.println s!"[ring] sum w = {rw.logWeight}"
  let e : Evidence := evs[0]!
  let event : RingEvent := { atStep := 0, kind := .leanSearch, evidenceId := e.id, payloadHash := "sha1-abc123", phi := rw.phi }
  let log := RingLog.appendEvent RingLog.empty event
  IO.println s!"[ring] events={log.events.size}"