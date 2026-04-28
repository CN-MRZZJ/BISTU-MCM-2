# Q1 Result Summary

## Fastest annual decline
| filter_no   |   annual_decline_rate |   avg_last_365d |   safety_margin_G |
|:------------|----------------------:|----------------:|------------------:|
| A10         |              -48.9523 |         71.1609 |           34.1609 |
| A8          |              -32.962  |         80.5448 |           43.5448 |
| A5          |              -31.6895 |         75.9567 |           38.9567 |

## Highest life risk
| filter_no   |   avg_last_365d |   safety_margin_G | risk_level   |
|:------------|----------------:|------------------:|:-------------|
| A2          |         67.0246 |           30.0246 | low          |
| A10         |         71.1609 |           34.1609 | low          |
| A5          |         75.9567 |           38.9567 | low          |

## Mean maintenance effect
- Mid maintenance mean delta: 16.7479
- Big maintenance mean delta: 13.7048

## Common regression terms
| term                   |     coef |   std_error |   t_value |       r2 |
|:-----------------------|---------:|------------:|----------:|---------:|
| season_sin             |  9.09679 |    0.168006 |  54.1457  | 0.845591 |
| season_cos             | -9.47295 |    0.151592 | -62.4897  | 0.845591 |
| recent_mid_maintenance |  1.21341 |    0.35977  |   3.37275 | 0.845591 |
| recent_big_maintenance | -1.65937 |    0.779699 |  -2.12822 | 0.845591 |
| cum_mid_maintenance    | 15.3996  |    0.385387 |  39.9588  | 0.845591 |
| cum_big_maintenance    | 15.9196  |    0.457721 |  34.7801  | 0.845591 |
