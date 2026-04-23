## Random problems thing for **YOUR KNOWLEDGE**

### 1. What are reward Hacking?
- When we train RL problems, sometimes model doesnt focus on the task and ratherfinds a way to maximize the reward function and gets max reward and focuses on that, rather than focusing on doing the task.

### 2. What is KL Diveregence Constraint? Why do we use it?
- KL divergence is nothing but the difference between two probability distributions.
- In RL, we use KL divergence constraint to prevent the model from diverging from the original policy/model. If we let it optimize freely, it will forget the task and go away from the original model's task, which we dont want and hence there exists this *KL Divergence loss*, which is added to total loss to keep it limited to what it is supposed to do.
- We do something like this - `total_reward = reward_score - β * KL(policy || reference)`, where `β` is the leash which keeps how much loss to keep.
- Very high beta = Model doesnt go too away from original model (very less learning)
- Very low beta = Model goes away ignoring KL diveregence  (model drifting)

### 3. Constitutional AI (Anthropic)
- Instead of having human labelling the preferences for Reinforcement learning training data, they use Claude itself to provide preferences for the data. Instead of RLHF, more like RLAIF (Reinforcement learning from AI Feedback).

### 4. Scalable Oversight Problem
- Real unsolved problem. If the model gets smarter then human at some field and they write some crazy proof, we dont have anyone to oversight it and say, YES THIS IS CORRECT. Because we dont know if its correct or not YET.

### 5. Read teaming
- Every major lab has a job to *Break* a model by doing adversial attacks on the mode by Prompt injections, jailbreaks, getting it to produce harmful content; etc. To see how bad the model is to such attacks by hackers or some people.
- Part of A10 Firewall system I think.

### 6. What is Alignment Tax?
- Dirty secret of the industry — RLHF makes models safer but also makes them dumber on benchmarks. There's a measurable performance drop on reasoning tasks after alignment. Companies manage this tradeoff constantly. Too much alignment = model refuses everything and becomes useless. Too little = model is capable but dangerous. Finding that line is literally a full time research problem.