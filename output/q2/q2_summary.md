# Q2 Result Summary

## Physical constraints
- Filter trend coefficients are constrained by beta_i <= 0.
- Cumulative maintenance coefficients are constrained by eta_M <= 0 and eta_B <= 0.
- Exponential maintenance recovery coefficients are constrained by gamma_M >= 0 and gamma_B >= 0.

## Selected maintenance decay
- Mid maintenance half-life: 180 days, lambda = 0.003851
- Big maintenance half-life: 180 days, lambda = 0.003851

## Shortest predicted remaining life
| filter_no   |   annual_decline_rate |   current_safety_margin_G | failure_date   |   remaining_years | status                |
|:------------|----------------------:|--------------------------:|:---------------|------------------:|:----------------------|
| A10         |              -68.3483 |                   34.1609 | 2026-11-03     |          0.564384 | failed_within_horizon |
| A5          |              -49.107  |                   38.9567 | 2027-08-12     |          1.33973  | failed_within_horizon |
| A8          |              -55.361  |                   43.5448 | 2027-08-17     |          1.35342  | failed_within_horizon |
| A7          |              -34.9258 |                   45.415  | 2027-11-27     |          1.63288  | failed_within_horizon |
| A9          |              -32.0608 |                   40.6188 | 2028-01-17     |          1.7726   | failed_within_horizon |

## Key model terms
| term                  |         coef |   std_error |       t_value |       r2 |
|:----------------------|-------------:|------------:|--------------:|---------:|
| season_sin            | 10.7739      |    0.163131 |  66.0446      | 0.846123 |
| season_cos            | -7.11364     |    0.150688 | -47.2078      | 0.846123 |
| decay_mid_maintenance | 11.5826      |    0.275039 |  42.1126      | 0.846123 |
| decay_big_maintenance | 13.1963      |    0.980648 |  13.4567      | 0.846123 |
| cum_mid_maintenance   | -1.96294e-25 |    0.396702 |  -4.94815e-25 | 0.846123 |
| cum_big_maintenance   | -5.91709e-26 |    0.907244 |  -6.52204e-26 | 0.846123 |
