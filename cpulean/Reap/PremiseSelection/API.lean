module
public meta import Lean.Data.Json
public meta import Reap.Options

public meta section

open Lean

/-- v1-1: premise selection 为纯数据契约；HTTP 由 driver/python 侧实现（跨设备通讯）。 -/
structure PremiseSelectionRequest where
  query : String
  numResults : Nat

structure PremiseSelectionResult where
  formalName : String
  formalStatement : String

def toJsonRequest (r : PremiseSelectionRequest) : Json :=
  Json.mkObj [("query", Json.str r.query), ("num_results", Json.num r.numResults)]

def toJsonResult (r : PremiseSelectionResult) : Json :=
  Json.mkObj [("formal_name", Json.str r.formalName),
              ("formal_statement", Json.str r.formalStatement)]

def fromJsonRequest (j : Json) : Except String PremiseSelectionRequest := do
  let obj ← match j.getObj? with | .ok v => pure v | .error e => throw e
  let q := match obj.get? "query" with
    | some v => match v.getStr? with | .ok s => s | .error _ => ""
    | none => ""
  let n := match obj.get? "num_results" with
    | some v => match v.getInt? with | .ok i => i.toNat | .error _ => 6
    | none => 6
  return { query := q, numResults := n }

instance : FromJson PremiseSelectionRequest where fromJson? j := fromJsonRequest j
instance : ToJson PremiseSelectionRequest where toJson r := toJsonRequest r
instance : ToJson PremiseSelectionResult where toJson r := toJsonResult r

structure PremiseSelectionClient where
  apiUrl : String

initialize cache :
  IO.Ref (Std.HashMap (String × Nat) (Array PremiseSelectionResult)) ← IO.mkRef {}

namespace PremiseSelectionClient

def mkRequest (s : String) (numResults : Nat := 6) : PremiseSelectionRequest :=
  { query := s, numResults := numResults }

def queryVia
    (req : PremiseSelectionRequest)
    (transport : PremiseSelectionRequest → IO (Array PremiseSelectionResult)) :
    IO (Array PremiseSelectionResult) := do
  match (← cache.get).get? (req.query, req.numResults) with
  | some results => return results
  | none => do
    let results ← transport req
    cache.modify fun m => m.insert (req.query, req.numResults) results
    return results

end PremiseSelectionClient

end
