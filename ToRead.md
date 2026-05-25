**Week 1 — RLHF & alignment foundations**
- DONE InstructGPT (Ouyang et al., 2022)
- DONE PPO (Schulman et al., 2017)
- DONE DPO (Rafailov et al., 2023)
- DONE Constitutional AI (Bai et al., 2022)
- DONE SafeRLHF (Dai et al., ICLR 2024) — NEW, directly addresses safety-aligned fine-tuning, exactly what JD asks for

**Week 2 — Jailbreaking foundations**
- DONE Jailbroken: How Does LLM Safety Training Fail? (Wei et al., 2023)
- DONE Universal and Transferable Adversarial Attacks / GCG (Zou et al., 2023)
- DONE PAIR: Jailbreaking Black Box LLMs in Twenty Queries (Chao et al., 2023)
- DONE TAP: Tree of Attacks with Pruning (Mehrotra et al., 2024)
- DONE AutoDAN-Turbo (Liu et al., ICLR 2025) — current SOTA automated red-teaming
- DONE Crescendo (Russinovich et al., Microsoft, 2024) — multi-turn attacks

**Week 3 — Prompt injection** (A10's product surface)
- DONE Not What You've Signed Up For (Greshake et al., 2023)
- DONE AgentDojo (Debenedetti et al., NeurIPS 2024)
- DONE Adaptive Attacks Break Defenses Against Indirect PI (Zhan et al., NAACL 2025)
- DONE Indirect Prompt Injections: Are Firewalls All You Need? (Shi et al., 2025)

**Week 3.5 — Bias & toxicity** (NEW, JD explicitly mentions "bias detection")
- DONE RealToxicityPrompts (Gehman et al., 2020) — canonical toxicity eval
- DONE BBQ: Bias Benchmark for QA (Parrish et al., 2022) — social bias eval, you'll likely use this
- SKIPPED DecodingTrust (Wang et al., NeurIPS 2023) — comprehensive trustworthiness eval, A10 will probably reference this

**Week 4 — Defenses, guardrails, evaluation frameworks** (THIS IS THE JOB)
- DONE Llama Guard 2 / Llama Guard 3 technical reports
- HarmBench (Mazeika et al., 2024)
- StrongREJECT (Souly et al., NeurIPS 2024)
- WildGuard (Han et al., NeurIPS 2024) — NEW, modern open-source safety classifier, direct comparable to what A10 builds
- SKIPPED OWASP Top 10 for LLM Applications 2025

**Week 4 side quest — tooling (1-2 days each, hands-on)** - CAN'T
- ACTIVITY Garak (NVIDIA) — run against an open model, document findings
- ACTIVITY PyRIT (Microsoft) — skim docs
- ACTIVITY Inspect (UK AISI) — modern eval framework
- DONE ACTIVITY Run AgentDojo locally — direct portfolio talking point

**Optional / stretch (only if ahead of schedule)** OPTIONAL
- Visual Adversarial Examples Jailbreak Aligned LLMs (Qi et al., 2024) — multimodal angle, JD mentions multimodal
- Latent Adversarial Training (Sheshadri et al., 2024) — frontier defense technique