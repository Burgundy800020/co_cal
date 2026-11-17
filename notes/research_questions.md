# Open research questions
- Which signal tells us the policy is unreliable: Q-value, Q-ensemble disagreement, something else?
- How to set the intervention threshold (cf. ThriftyDAgger) under a fixed expert budget?
- How many expert steps per takeover, and when to hand control back?
- Fine-tuning loss: TD3 + online BC on expert steps; how to weight the two?
- Expert: infinite-budget online RL on top of the offline model (mimics a human who can recover).
- Evaluation: episode return vs. number of online terminations; compare to offline-to-online TD3 and DAgger.

## Online BC reg 
- Which ones to bc? replay, offline dataset, both?
- BC lasts for entire training run or only the first phase?