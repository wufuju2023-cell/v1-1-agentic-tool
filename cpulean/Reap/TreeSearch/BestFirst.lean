module
public meta import Reap.TreeSearch.Basic

namespace TreeSearch

public meta section
variable {m : Type → Type} [Monad m] {σ : Type}

def BestFirst.defaultMaxNodes : Nat := 64

/-- 线性 max：Option 哨兵，避免下标访问与 Inhabited 需求。 -/
private def maxPair (h : Array (Float × σ)) : Option (Float × σ) :=
  h.foldl
      (fun acc (e : Float × σ) =>
        match acc with
        | none => some e
        | some p => some (if e.1 > p.1 then e else p))
      (none : Option (Float × σ))

def bestFirstSearch [DecidableEq σ] (priority : σ → m Float)
    (isTerminal : σ → m Bool) (expand : σ → m (Array σ)) (start : σ)
    (maxNodes : Nat := BestFirst.defaultMaxNodes)
    : m (Option σ) := do
  let mut frontier : Array (Float × σ) := #[(0.0, start)]
  let mut visited := 0

  while visited < maxNodes do
    if frontier.size = 0 then
      return none
    let some (p, s) := maxPair frontier | return none
    visited := visited + 1
    if ← isTerminal s then
      return some s
    frontier := frontier.filter (fun e => e.2 ≠ s)
    for s' in ← expand s do
      if !frontier.any (fun e => e.2 == s') then
        frontier := frontier.push (← priority s', s')
  return none

end

end TreeSearch
