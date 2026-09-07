/-
ReapAgentic — v1-1-agentic 核心模块

在经典 REAP 树搜索之外，定义"证据环"（agentic evidence ring）的数据与组合规则：
树内先验 π 由 GPU policy 提供，树外证据 R（opencode / link search / lean search / tool call）
以贝叶斯重加权 π' (s, R) ∝ π (s) * P (R | t, s) 的方式改变先验。

纯 Lean 标准库 + Batteries（不依赖 mathlib API），可独立编译、可被 Test 驱动使用。

本文件只管"数据与组合律"，不做 IO；IO（HTTP/工具调用）由 driver 层实现。
-/
import Lean.Data.Json
import Reap.Options
open Lean

namespace Reap.Agentic

/-- 证据来源类别：对应 agentic 工具环的四种通道 + 人工补注 -/
inductive EvidenceKind where
  | linkSearch   -- 网页/仓库链接搜索（opencode web tool）
  | leanSearch   -- Lean 库/定理搜索（PreMiseSelection 风格）
  | fileSearch   -- repo 文件导航读取
  | toolCall     -- 计算/验证工具（lake env lean、arity 检查等）
  | manual
  deriving DecidableEq, Repr

instance : ToString EvidenceKind where
  toString
  | .linkSearch => "linkSearch"
  | .leanSearch => "leanSearch"
  | .fileSearch => "fileSearch"
  | .toolCall => "toolCall"
  | .manual => "manual"

instance : ToJson EvidenceKind where
  toJson k := json% $(toString k)

instance : FromJson EvidenceKind where
  fromJson? j :=
    match j.getStr? with
    | .ok s =>
      .ok <| match s with
        | "linkSearch" => .linkSearch
        | "leanSearch" => .leanSearch
        | "fileSearch" => .fileSearch
        | "toolCall" => .toolCall
        | _ => .manual
    | .error _ => .error "not a string"

/-- 一条证据：来源、可信权重与原始载荷 -/
structure Evidence where
  id : String
  kind : EvidenceKind
  /-- weight in [0,1]：0=无关，1=决定性 -/
  weight : Float
  payload : Json
  sourceDesc : String

instance : ToJson Evidence where
  toJson e :=
    json% {
      "id" : $(e.id),
      "kind" : $(e.kind),
      "weight" : $(e.weight.toString),
      "payload" : $(e.payload),
      "sourceDesc" : $(e.sourceDesc)
    }

/-- 证据的有效聚合：对数域求和后经饱和压缩到 [0,1]：
   ℓ = Σ_k w_k ;  φ = tanh(ℓ)  (饱和，避免单条"决定性"证据无限主导) -/
def evidenceComposite (ev : Array Evidence) : Float :=
  let ℓ := ev.foldl (fun acc e => acc + e.weight) 0.0
  Float.tanh ℓ

/-- 贝叶斯式重加权结果：π' 的缩放因子 φ 与一个可验证的说明书 -/
structure PriorReweight where
  /-- composite strength in [0,1] -/
  phi : Float
  /-- total log weight ℓ = Σ w -/
  logWeight : Float
  /-- 支持端口的证据列表（可回放审计） -/
  evidence : Array Evidence

def reweightBayes (ev : Array Evidence) : PriorReweight :=
  { phi := evidenceComposite ev, logWeight := ev.foldl (fun acc e => acc + e.weight) 0.0,
    evidence := ev }

/-- 证据环运行中的单条事件（供 RolloutSink / Verdict 审计） -/
structure RingEvent where
  atStep : Nat
  kind : EvidenceKind
  evidenceId : String
  payloadHash : String
  phi : Float
  deriving Repr

instance : ToJson RingEvent where
  toJson r :=
    json% {
      "atStep" : $(r.atStep),
      "kind" : $(r.kind),
      "evidenceId" : $(r.evidenceId),
      "payloadHash" : $(r.payloadHash),
      "phi" : $(r.phi.toString)
    }

/-- 会话级审计日志（写入 RolloutSink 的 agentic 扩展段） -/
structure RingLog where
  session : String
  events : Array RingEvent

namespace RingLog
def empty : RingLog := { session := "", events := #[] }
def appendEvent (rl : RingLog) (e : RingEvent) : RingLog :=
  { rl with events := rl.events.push e }
end RingLog

/-- v1-1 的三个插槽：证据来源、重加权与树搜索的契约动作。
   driver 层实现 `runEvidence`: 具体调用 opencode / link search / lean search 工具并产出 Evidence。 -/
structure V11Contract where
  /-- 每个 course 最多证据查询次数 -/
  maxEvidenceQueries : Nat := 24
  /-- 温度覆盖：t = 1 表示保持模型温度不变 -/
  temperature : Nat := 100
  /-- 先验重加权衰减（evidence 过期窗口,以 step 计） -/
  decayAfterSteps : Nat := 8

end Reap.Agentic
