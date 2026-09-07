module
/-
v1-1 Generator：无 OpenAIClient 依赖版。

公开契约不变：TacticGenerator / getClient / mkPrompt / generatePolicyFromPrompt /
generateValueFromPrompt / generatePolicyValue / ValueResult / getRelatedTheorems。

v1-1 跨界语义：transport 参数化（CoreM 函数），driver 注入真实 GPU/证据环后端；
未注入时走默认 stub（空/低分），保证纯编译可跑。
-/
public meta import Lean.Elab.Task
public meta import Reap.Options
public meta import Reap.PremiseSelection.Syntax

public meta section

open Lean

structure TacticGenerator where
  polT : String → Nat → CoreM (Array (String × Float))
  valT : String → CoreM Float

namespace TacticGenerator

initialize policyTransportRef :
  IO.Ref (String → Nat → CoreM (Array (String × Float))) ← IO.mkRef (fun _ _ => pure #[])

initialize valueTransportRef :
  IO.Ref (String → CoreM Float) ← IO.mkRef (fun _ => pure (-1000.0))

def setPolicyTransport (f : String → Nat → CoreM (Array (String × Float))) : IO Unit :=
  policyTransportRef.modify fun _ => f

def setValueTransport (f : String → CoreM Float) : IO Unit :=
  valueTransportRef.modify fun _ => f

end TacticGenerator

def stripThinkingPrefix (s : String) : String :=
  let parts := s.splitOn "<｜end▁of▁thinking｜>"
  if parts.length > 1 then String.intercalate "<｜end▁of▁thinking｜>" (parts.drop 1) else s

/-- 兼容 OpenAI chat/text 形态：解析 choices → text strings（各配权重 1.0，
   logprob 说明由 GPU 契约 v2 提供；v1-1 使用 `p × Φ` 组合时以 driver 为准）。 -/
def parseChatResponseOpenAI (res : String) : Array (String × Float) :=
  match Json.parse (stripThinkingPrefix res) with
  | .ok j =>
    match j.getObj? with
    | .ok obj =>
      match obj.get? "choices" with
      | some arrJ =>
        match arrJ.getArr? with
        | .ok arr =>
          arr.filterMap (fun c =>
            match c.getObj? with
            | .ok co =>
              match co.get? "text" with
              | some t => match t.getStr? with
                | .ok s => some (s, 1.0)
                | .error _ => none
              | none => none
            | .error _ => none)
        | .error _ => #[]
      | none => #[]
    | .error _ => #[]
  | .error _ => #[]

def mkRelatedTheorem (ps : PremiseSelectionResult) : String :=
  "lemma " ++ ps.formalName ++ " : " ++ ps.formalStatement

def mkPrompt (tacticState : String) (relatedTheorems : Array PremiseSelectionResult) : String :=
  let ps := relatedTheorems.toList.map mkRelatedTheorem |> String.intercalate "\n"
  "State:\n" ++ tacticState ++ "\n\nRelated theorems:\n" ++ ps

structure ValueResult where
  score : Float

def parseValueScore (text : String) : Float :=
  match Json.parse text with
  | .ok j =>
    match j.getObj? with
    | .ok obj =>
      match obj.get? "score" with
      | some v => match v.getNum? with
        | .ok n => n.toFloat
        | .error _ => 0.0
      | none => 0.0
    | .error _ => 0.0
  | .error _ => 0.0

def getClient : CoreM TacticGenerator := do
  let polT ← TacticGenerator.policyTransportRef.get
  let valT ← TacticGenerator.valueTransportRef.get
  return { polT := polT, valT := valT }

def getRelatedTheorems (mvarIds : List MVarId) (ppGoal : String)
    (opts : Options) : MetaM (Array PremiseSelectionResult) := do
  selectPremisesForGoals mvarIds (reap.num_premises.get opts)

def generatePolicyFromPrompt (generator : TacticGenerator) (opts : Options)
    (ppGoal : String) (relatedTheorems : Array PremiseSelectionResult) (prompt : String) :
    CoreM (Array (String × Float)) := do
  let n := reap.num_samples.get opts
  let results ← generator.polT prompt n
  return results.filter fun (s, _) => !s.trim.isEmpty

def generateValueFromPrompt (generator : TacticGenerator) (opts : Options)
    (ppGoal : String) (relatedTheorems : Array PremiseSelectionResult) (prompt : String) :
    CoreM Float := do
  generator.valT prompt

def Meta.ppProofState (mvarIds : List MVarId) : MetaM Format := do
  return Std.Format.joinSep (← mvarIds.mapM (Meta.ppGoal)) "\n".toFormat

/-- policy+value 并行生成（契约与上游一致） -/
def generatePolicyValueImpl (mvarIds : List MVarId) :
    MetaM <| Float × Array (String × Array PremiseSelectionResult × Float) := do
  let opts ← getOptions
  let generator ← getClient
  let ppProofState := toString (← Meta.ppProofState mvarIds)
  let relatedTheorems ← getRelatedTheorems mvarIds ppProofState opts
  let prompt := mkPrompt ppProofState relatedTheorems
  let (_, valueTask) ← Lean.Core.CoreM.asTask <|
    generateValueFromPrompt generator opts ppProofState relatedTheorems prompt
  let (_, policyTask) ← Lean.Core.CoreM.asTask <|
    generatePolicyFromPrompt generator opts ppProofState relatedTheorems prompt
  let value ← valueTask.get
  let tactics ← policyTask.get
  return (value, tactics.map fun (x, y) => (x, relatedTheorems, y))

namespace TacticGenerator

/-- 兼容别名：上游调用点 `TacticGenerator.generatePolicyValue`。 -/
def generatePolicyValue (mvarIds : List MVarId) :
    MetaM <| Float × Array (String × Array PremiseSelectionResult × Float) :=
  generatePolicyValueImpl mvarIds

end TacticGenerator

end
