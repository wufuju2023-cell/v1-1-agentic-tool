# v1-1-agentic-tool

v1-1-agentic 工具链（Lean CPU 运行时 + driver/smoke），**基线对齐容器环境**：

| 层 | 版本 | 状态 |
|---|---|---|
| Lean 工具链 | **leanprover/lean4:v4.28.0**（`cpulean/lean-toolchain`） | 主链编译通过（Lean 直编验证） |
| mathlib | `leanprover-community/mathlib4 rev v4.28.0`（lake 依赖） | 与容器评估环境（existing-project-Lean-4.28）对齐 |
| 数据环境 | Mathlib 4.19（leantree dataset 发布环境） | 训练数据契约 |

## 目录

```
cpulean/                  # Lean CPU 运行时（v4.28 兼容）
  lean-toolchain          #   v4.28.0
  lakefile.toml           #   require mathlib (git v4.28.0)
  Reap/
    Agentic.lean          #   v1-1 证据环：Evidence/RingLog/reweightBayes/V11Contract
    TreeSearch/{Basic,MCTS,BestFirst}.lean
    Tactic/{State,Step,Generator,Syntax,TreeSearch,WallClock}.lean
    PremiseSelection/{API,Syntax}.lean
    Training/{Verdict,RolloutSink}.lean
driver/                   # Python transport（mock/local-opencode/remote 三档）
smoke/                    # problems.json 3 题 + v1_smoke.py
tests/
```

## 本地验证（无 mathlib 大编译）

```bash
cd cpulean
OUT=$(pwd)/.build/olean && mkdir -p "$OUT"/{Reap/Tactic,Reap/TreeSearch,Reap/PremiseSelection,Reap/Training}
export LEAN_PATH="$OUT"
lean -o "$OUT/Reap/Options.olean" Reap/Options.lean
lean -o "$OUT/Reap/Agentic.lean.olean" Reap/Agentic.lean    # 依依赖序逐模块
```

## 4.28 兼容要点（相对 4.33 开发的差异）

- `[i]?`/`[i]!` 数组语法、`Json` API 以 v4.28 为准（`json%` 宏、`fromJson?` 字段）
- 移除 Batteries 硬依赖（BestFirst 线性堆、MCTS 纯 Std）
- `Float.inf/.huge` 不存在 → 1.0e300 哨兵
- `getTypeCleanup` → `getType`

## 关键约定（同 v1-1）

- 树内采样=GPU policy（RULE-0）；Agent 证据环=树外证据；`phi = tanh(Σ w)` 贝叶斯重加权
- Lean 零 HTTP；跨设备经 driver transport 注入（`setPolicyTransport/setValueTransport/queryVia`）
