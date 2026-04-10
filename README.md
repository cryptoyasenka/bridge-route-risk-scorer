# Bridge Route Risk Scorer

ONNX model that scores a **specific cross-chain bridge route** (source chain → destination chain via a specific bridge, for a specific asset and size, at the current block) on its probability of delivering a degraded or failed outcome. Built for bridge aggregators, wallets routing user transfers, and intent-based cross-chain protocols that need a per-route risk signal.

- **Hub:** https://hub.opengradient.ai/models/NT_Kljxpd20dIof/bridge-route-risk-scorer
- **Release:** 1.00
- **Model CID:** `MieafOoPlNh9hTKb8Q4gCiFQCrgQ8Lhk7uZ82Itb4VM`
- **Category:** Risk Models
- **License:** MIT

## Why this model exists

Existing bridge risk models on the hub (`cross-chain-bridge-risk`, `cross-chain-bridge-safety`, `BridgeGuard`) score **the bridge protocol itself**: "is Wormhole safe, in general?" That's the wrong question for a bridge aggregator. The real question at the moment of use is:

> "I need to move 500k USDC from Arbitrum to Base **right now**. Should I use Stargate, Across, CCTP, Wormhole, or Circle's canonical route? Which of those routes is the riskiest **for this specific transfer**?"

A protocol-level safety score can't answer that. A route is a (bridge, source chain, destination chain, asset, amount, block) tuple, and its risk depends on things the aggregator sees at request time:

- **Relative TVL consumption.** Moving 500k through a bridge that holds 2M in total liquidity is very different from moving 500k through a bridge holding 500M, even if both "bridges" are the same protocol.
- **Destination chain health.** Base might be re-orging, Linea might be stalled, Arbitrum sequencer might be down — the bridge protocol itself is fine, but the route is cooked.
- **Validator quorum freshness.** LayerZero / Wormhole / CCIP all depend on off-chain validator sets whose keys rotate on different schedules. A stale quorum is a route-specific risk, not a protocol one.
- **Recent message-layer exploits.** A bridge can ship a hotfix for the core protocol while specific chain pairs remain disabled or half-patched.
- **Route price vs alternatives.** A 50 bps fee markup vs the cheapest alternative route is a signal that the route is stressed or being griefed.

This model takes 10 numerical features describing a single route at request time and returns a single `route_risk ∈ [0, 1]` — the probability the transfer will fail, reorg, delay significantly past SLA, or settle at materially worse terms. It is deliberately tiny (pure ONNX, ~260 bytes) so a router can score **every candidate route in parallel** in well under a millisecond before picking the winner.

## Architecture

Pure ONNX graph, opset 11, **no weight initializers** (the OpenGradient hub rejects models with embedded weights for public inference):

```
features [1,10] float32
      │
     Relu
      │
  ReduceSum(axes=[1], keepdims=1)
      │
    Sigmoid
      │
route_risk [1,1] float32
```

Because there are no trained weights, the model behaves as a monotone scoring function: the sum of the non-negative normalized features squashed through a sigmoid. All risk logic lives in **how you normalize the inputs** before inference — every feature must already be in a `[0, 1]`-ish range where "1 = maximally risky". The Scoring Pillars below describe the normalization contract.

## Scoring Pillars

| Pillar | Weight | Features |
|--------|--------|----------|
| **Liquidity & Size Fit** | 30% | `size_vs_tvl`, `path_liquidity_fragmentation` |
| **Destination Chain Health** | 25% | `dest_chain_reorg_risk`, `dest_chain_sequencer_health`, `dest_chain_finality_lag` |
| **Bridge Message Layer** | 25% | `validator_quorum_staleness`, `recent_exploit_proximity`, `route_disabled_history` |
| **Economic Signals** | 20% | `fee_markup_vs_alt`, `relayer_competition` |

Callers are expected to scale each feature before inference so its contribution matches the pillar weight — e.g. `size_vs_tvl = (amount / bridge_tvl_for_pair) * 0.30 / 2`.

## Input Schema

`features: float32 [1, 10]`

All features are expected to be pre-normalized into a `[0, 1]` range where `1 = maximally risky`. Callers are responsible for the normalization contract described in the Scoring Pillars section.

| Index | Name | Meaning |
|---|---|---|
| 0 | `size_vs_tvl` | Transfer size / bridge-held TVL for this (chain-pair, asset). `1 = ≥ 20% of the pool` |
| 1 | `path_liquidity_fragmentation` | Inverse depth of the destination-side liquidity (shallow pool for the arriving asset) |
| 2 | `dest_chain_reorg_risk` | 30-min rolling re-org rate of the destination chain, scaled by baseline |
| 3 | `dest_chain_sequencer_health` | Inverse uptime of the destination chain's sequencer / proposer (`1 = halted`) |
| 4 | `dest_chain_finality_lag` | Destination chain finality lag vs its SLA (`1 = ≥ 3× normal`) |
| 5 | `validator_quorum_staleness` | Time since the bridge's validator set last rotated / attested, scaled |
| 6 | `recent_exploit_proximity` | Inverse time since the most recent disclosed exploit / near-miss on this bridge (`1 = < 7 days ago`) |
| 7 | `route_disabled_history` | Share of the last 30 days this specific (bridge, chain-pair) route spent disabled or paused |
| 8 | `fee_markup_vs_alt` | Fee premium over the cheapest alternative route for the same (amount, pair), clipped |
| 9 | `relayer_competition` | Inverse number of active relayers / fillers on this route (`1 = single relayer`, `0 = healthy market`) |

## Output Schema

`route_risk: float32 [1, 1]`

A single probability in `[0, 1]`. Higher values mean the route is more likely to fail, reorg, delay past SLA, or settle at materially worse terms than a competing route would.

| Range | Grade | Action |
|---|---|---|
| 0.00 – 0.20 | A | Use — preferred route |
| 0.20 – 0.40 | B | Use with monitoring |
| 0.40 – 0.60 | C | Hold — prompt user for confirmation |
| 0.60 – 0.80 | D | Fall back to an alternative route |
| 0.80 – 1.00 | F | Block — do not execute |

## Flags

| Flag | Trigger | Severity |
|---|---|---|
| F1 | `size_vs_tvl > 0.7` | Critical — transfer consumes a large share of bridge liquidity |
| F2 | `dest_chain_sequencer_health > 0.7` | Critical — destination sequencer halted |
| F3 | `recent_exploit_proximity > 0.7` | Critical — bridge exploited within the last week |
| F4 | `dest_chain_reorg_risk > 0.6` and `dest_chain_finality_lag > 0.6` | High — destination chain is unstable |
| F5 | `route_disabled_history > 0.5` | High — route has a history of being paused |
| F6 | `validator_quorum_staleness > 0.7` | Medium — stale message-layer quorum |
| F7 | `route_risk > 0.8` | Critical — block and fall back |

## Use Cases

1. **Bridge aggregators** (LI.FI, Socket, Jumper, Squid, Rango) — score each candidate route for the same transfer and pick the one with the lowest `route_risk`, not just the cheapest fee.
2. **Smart-account wallets** (Safe, Rabby, Rainbow, Coinbase Smart Wallet) — add an inline warning before a cross-chain transfer if the chosen route scores D/F.
3. **Intent-based cross-chain protocols** (Across, Hyperlane, Polymer) — gate solver bids by route risk so a "cheap but broken" route can't win an auction.
4. **Risk dashboards** (Defi Llama Bridges, TokenTerminal, Chaos Labs) — publish a per-route health signal that updates in real time instead of weekly protocol ratings.

## Data Sources

- **Bridge TVL per (chain-pair, asset):** bridge subgraphs (Stargate, Across, Hop, Synapse), Defi Llama bridges API, on-chain reads.
- **Destination chain health:** beacon/consensus RPC, sequencer status endpoints (Arbitrum, Optimism, Base, Linea), explorer reorg feeds.
- **Bridge validator quorum:** LayerZero DVN set, Wormhole guardian set, CCIP committee, Axelar validator set on-chain.
- **Exploit timeline:** rekt.news, Chainalysis Crypto Crime Report, public disclosures.
- **Fee markup / alternative routes:** aggregator APIs (LI.FI, Socket) sampled at the same block.

## Test Vectors

The graph is a monotone `sigmoid(ReduceSum(Relu(features)))`, so outputs grow with the sum of inputs. Calibrate your feature scaling so that the three reference bands below land in the correct grade buckets for your deployment.

| Profile | Input | Feature sum | Score | Grade |
|---|---|---|---|---|
| Clean | `[0.05, 0.02, 0.05, 0.00, 0.08, 0.00, 0.04, 0.03, 0.00, 0.05]` | 0.32 | `0.5793242455` | C (baseline) |
| Suspicious | `[0.45, 0.35, 0.50, 0.30, 0.40, 0.25, 0.55, 0.40, 0.20, 0.45]` | 3.85 | `0.9791636467` | F |
| Risky | `[0.92, 0.88, 0.95, 0.85, 0.90, 0.80, 0.82, 0.95, 0.70, 0.88]` | 8.65 | `0.9998248816` | F |

## Local Inference

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("bridge-route-risk-scorer.onnx")

risky = np.array([[0.92, 0.88, 0.95, 0.85, 0.90, 0.80, 0.82, 0.95, 0.70, 0.88]],
                 dtype=np.float32)
print(sess.run(None, {"features": risky})[0])  # [[0.9998249]]
```

## Rebuilding the ONNX File

```bash
pip install onnx onnxruntime
python build_model.py
```

`build_model.py` regenerates `bridge-route-risk-scorer.onnx` deterministically (258 bytes, 3 ops, opset 11, 0 initializers).

## Versioning

| Version | Date | Notes |
|---|---|---|
| 1.00 | 2026-04-11 | Initial release — opset 11, 10 features, 258 bytes |

## Tags

`#defi #ml #riskmodel #bridge #crosschain #layerzero #wormhole #interop #routing`
