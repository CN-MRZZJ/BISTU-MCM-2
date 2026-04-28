# Q4 Result Summary

Share of tested combined scenarios where the Q3 plan remains exactly optimal: 18.08%
Maximum annual-cost penalty of keeping the Q3 plan: 3.5650%

## One-way sensitivity
| scenario      |   same_filter_count |   base_plan_total_annual_cost |   reoptimized_total_annual_cost |   annual_cost_penalty_pct |
|:--------------|--------------------:|------------------------------:|--------------------------------:|--------------------------:|
| purchase_x0.7 |                   9 |                       1104.9  |                         1103.68 |               0.110501    |
| mid_x0.7      |                   9 |                       1328.55 |                         1326.58 |               0.148411    |
| big_x0.7      |                   5 |                       1359.32 |                         1351.4  |               0.585667    |
| purchase_x0.8 |                   9 |                       1204.84 |                         1204.11 |               0.0604988   |
| mid_x0.8      |                  10 |                       1353.94 |                         1353.94 |               1.67934e-14 |
| big_x0.8      |                   9 |                       1374.46 |                         1373.71 |               0.0544264   |
| purchase_x0.9 |                   9 |                       1304.78 |                         1304.55 |               0.0181954   |
| mid_x0.9      |                  10 |                       1379.34 |                         1379.34 |               0           |
| big_x0.9      |                  10 |                       1389.59 |                         1389.59 |               1.63626e-14 |
| purchase_x1.0 |                  10 |                       1404.73 |                         1404.73 |              -1.61863e-14 |
| mid_x1.0      |                  10 |                       1404.73 |                         1404.73 |              -1.61863e-14 |
| big_x1.0      |                  10 |                       1404.73 |                         1404.73 |              -1.61863e-14 |
| purchase_x1.1 |                  10 |                       1504.67 |                         1504.67 |               0           |
| mid_x1.1      |                  10 |                       1430.12 |                         1430.12 |               0           |
| big_x1.1      |                   9 |                       1419.86 |                         1419.65 |               0.0150135   |
| purchase_x1.2 |                   8 |                       1604.62 |                         1600.62 |               0.249348    |
| mid_x1.2      |                  10 |                       1455.51 |                         1455.51 |               0           |
| big_x1.2      |                   9 |                       1435    |                         1434.32 |               0.0474104   |
| purchase_x1.3 |                   8 |                       1704.56 |                         1693.2  |               0.670698    |
| mid_x1.3      |                  10 |                       1480.91 |                         1480.91 |               0           |
| big_x1.3      |                   9 |                       1450.14 |                         1448.99 |               0.0791513   |

## Worst combined scenarios
|   purchase_factor |   mid_factor |   big_factor |   same_filter_count |   annual_cost_penalty_pct |
|------------------:|-------------:|-------------:|--------------------:|--------------------------:|
|               1.3 |          0.7 |          0.7 |                   3 |                   3.56496 |
|               1.3 |          0.8 |          0.7 |                   3 |                   3.27739 |
|               1.3 |          0.9 |          0.7 |                   4 |                   3.0417  |
|               1.2 |          0.7 |          0.7 |                   3 |                   2.83358 |
|               1.3 |          1   |          0.7 |                   4 |                   2.82025 |

## Representative plan switches
| scenario      | filter_no   |   base_mid_interval_days |   base_big_interval_days |   mid_interval_days |   big_interval_days |
|:--------------|:------------|-------------------------:|-------------------------:|--------------------:|--------------------:|
| purchase_low  | A5          |                       30 |                      180 |                  30 |                 240 |
| purchase_high | A2          |                       30 |                      365 |                  30 |                 150 |
| purchase_high | A8          |                      120 |                      300 |                  30 |                  90 |
| mid_low       | A8          |                      120 |                      300 |                  30 |                 150 |
| big_low       | A2          |                       30 |                      365 |                  30 |                 150 |
| big_low       | A3          |                       30 |                      365 |                  30 |                 240 |
| big_low       | A5          |                       30 |                      180 |                  30 |                 120 |
| big_low       | A7          |                       30 |                      300 |                  30 |                 150 |
| big_low       | A8          |                      120 |                      300 |                  30 |                  90 |
| big_high      | A5          |                       30 |                      180 |                  30 |                 240 |
