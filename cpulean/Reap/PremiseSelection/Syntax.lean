module
public meta import Lean.LibrarySuggestions.Basic
public meta import Reap.Options
public meta import Reap.PremiseSelection.API
public meta section

open Lean
open Lean.LibrarySuggestions

/-- v1-1: transport 由 driver 注入（跨设备实现）；此处为编译期 stub。 -/
private def premisesTransportStub (_ : PremiseSelectionRequest) : IO (Array PremiseSelectionResult) :=
  pure #[]

def reapSelector : Selector := ppSelector fun ppStr config => do
  let rs ← PremiseSelectionClient.queryVia
    { query := ppStr, numResults := config.maxSuggestions } premisesTransportStub
  let suggestions := rs.map fun x => {
    name := x.formalName.toName
    score := 1.0
  }
  suggestions.filterM fun s => config.filter s.name

def suggestionToPremiseSelectionResult? (suggestion : Suggestion) :
    MetaM (Option PremiseSelectionResult) := do
  try
    let decl ← getConstInfo suggestion.name
    let statement ← Meta.ppExpr decl.type
    return some {
      formalName := toString suggestion.name
      formalStatement := toString statement
    }
  catch _ =>
    return none

def selectPremisesForGoals (mvarIds : List MVarId) (maxSuggestions : Nat) :
    MetaM (Array PremiseSelectionResult) := do
  let mut seen : NameSet := {}
  let mut premises := #[]
  for mvarId in mvarIds do
    if premises.size >= maxSuggestions then
      break
    let suggestions ← Lean.LibrarySuggestions.select mvarId { maxSuggestions := maxSuggestions }
    for suggestion in suggestions do
      if premises.size >= maxSuggestions then
        break
      unless seen.contains suggestion.name do
        match ← suggestionToPremiseSelectionResult? suggestion with
        | some premise =>
            seen := seen.insert suggestion.name
            premises := premises.push premise
        | none =>
            pure ()
  return premises
