# Archive-only model results

Samples: 7,486 across ['archive_165516', 'archive_173729']

| features   | split                 |     r2 |   rmse |   mae |
|:-----------|:----------------------|-------:|-------:|------:|
| raw256     | within:archive_165516 |  0.626 |   8.65 |  6.03 |
| raw256     | within:archive_173729 |  0.758 |  10.1  |  8.12 |
| sqrt256    | within:archive_165516 |  0.592 |   9.03 |  6.52 |
| sqrt256    | within:archive_173729 |  0.751 |  10.22 |  8.97 |
| total      | within:archive_165516 |  0.819 |   6.01 |  4.75 |
| total      | within:archive_173729 |  0.327 |  16.82 | 14.25 |
| raw256     | leave-one-session-out | -0.913 |  22.6  | 18.22 |
| sqrt256    | leave-one-session-out | -1.072 |  23.52 | 19.26 |
| total      | leave-one-session-out |  0.514 |  11.39 | 10.38 |
