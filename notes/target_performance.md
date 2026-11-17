# Performance Targets

## Policy Performance
Recall: offline bc+rl on medium level behavior policy.
Then online rl with limited expert level intervention.

After offline phase, want (1): suboptimal expected return
(2): some episodes with very high return. This means
the policy can do great sometimes but is very inconsistent.

After online phase: want significant gain over the 
offline policy, using only a small number of 
online terminations. Does not need expert level. 
But great if it can.