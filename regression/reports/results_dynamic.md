# YL_grasp_dynamic split-trial model results

Samples: 25,175 across 8 trials (YL_dynamic_01, YL_dynamic_02, YL_dynamic_03, YL_dynamic_04, YL_dynamic_05, YL_dynamic_06, YL_dynamic_07, YL_dynamic_08)

| features   | split               |    r2 |   pearson_r |   rmse |   mae |
|:-----------|:--------------------|------:|------------:|-------:|------:|
| raw256     | leave-one-trial-out | 0.588 |       0.81  |  11.82 |  7.84 |
| sqrt256    | leave-one-trial-out | 0.446 |       0.746 |  13.7  |  9.51 |
| total      | leave-one-trial-out | 0.417 |       0.647 |  14.06 | 11.1  |
| scalars5   | leave-one-trial-out | 0.545 |       0.75  |  12.41 |  9.04 |
