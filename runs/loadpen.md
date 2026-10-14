# Expert slots on the busiest rank, over the mean, against stock routing

Qwen3-30B-A3B in-engine, `EPS=4,8,16 PENS=0,1,2,4,8`. Negative is a flatter spread than stock routing. Dispatch volume and busiest-rank receipts do not appear because they are bit-identical across every penalty in the row.

| ep | keep | `load_penalty=0` | `load_penalty=1` | `load_penalty=2` | `load_penalty=4` | `load_penalty=8` |
|---|---|---|---|---|---|---|
| 4 | 6 | +4.2% | -7.0% | -7.0% | -7.0% | -7.0% |
| 4 | 4 | +13.5% | +3.0% | +2.9% | +2.9% | +2.9% |
| 4 | 2 | +36.9% | +31.9% | +31.8% | +31.8% | +31.8% |
| 8 | 6 | +9.7% | -1.8% | -2.2% | -2.2% | -2.2% |
| 8 | 4 | +26.4% | +14.0% | +13.4% | +13.4% | +13.4% |
| 8 | 2 | +61.8% | +55.5% | +55.4% | +55.4% | +55.4% |
| 16 | 6 | +19.1% | +7.3% | +6.7% | +6.6% | +6.6% |
| 16 | 4 | +43.6% | +25.1% | +23.3% | +23.3% | +23.3% |
| 16 | 2 | +88.9% | +79.6% | +78.9% | +78.9% | +78.9% |
