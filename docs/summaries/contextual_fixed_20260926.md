# Contextual Enrichment Summary — Fixed Strategy (2026-09-26)

## Run Command

```bash
caffeinate -i uv run python main.py --strategy fixed --contextual --chunk-only
```

## Configuration

- **Strategy**: Fixed (chunk_size=500, overlap=50)
- **Contextual model**: claude-haiku-4-5-20251001
- **Dataset**: QuALITY (20 documents, 1,095 chunks)
- **Mode**: chunk-only (no evaluation)

## Prompt Cache Performance

| Document | Chunks | Cache Hit % | Input Tokens | Output Tokens |
|----------|--------|-------------|--------------|---------------|
| 20012    | 50     | 95%         | 8,810        | 4,790         |
| 20015    | 27     | 0%          | 90,770       | 2,643         |
| 20046    | 25     | 0%          | 67,598       | 2,012         |
| 20061    | 26     | 0%          | 80,821       | 2,245         |
| 24247    | 65     | 96%         | 14,340       | 5,824         |
| 24278    | 59     | 96%         | 12,122       | 4,683         |
| 24517    | 41     | 94%         | 8,682        | 3,707         |
| 51053    | 65     | 96%         | 13,478       | 6,210         |
| 51129    | 71     | 96%         | 14,445       | 6,572         |
| 51286    | 67     | 96%         | 13,775       | 5,311         |
| 51296    | 75     | 97%         | 15,612       | 6,007         |
| 51657    | 61     | 94%         | 12,674       | 5,707         |
| 55933    | 57     | 95%         | 11,920       | 4,519         |
| 61499    | 66     | 95%         | 13,444       | 5,017         |
| 62198    | 59     | 96%         | 11,945       | 5,119         |
| 62569    | 67     | 96%         | 14,444       | 5,840         |
| 62619    | 63     | 96%         | 13,023       | 5,268         |
| 63527    | 67     | 96%         | 13,976       | 5,877         |
| 99902    | 51     | 95%         | 10,154       | 4,133         |
| 99910    | 33     | 0%          | 113,317      | 2,265         |

**17/20 documents** achieved 94-97% cache hit ratios.
**3 documents** (20015, 20046, 20061, 99910) got 0% — their text was under Haiku's 4,096-token cache minimum.

## Token Totals

| Type | Tokens |
|------|--------|
| Prompt cache reads | 7,520,429 |
| Prompt cache writes | 137,656 |
| Uncached input | 555,350 |
| Output | 93,749 |

## Estimated Cost

| Component | Tokens | Rate (Haiku) | Cost |
|-----------|--------|--------------|------|
| Cache reads | 7,520,429 | $0.10/MTok | $0.75 |
| Cache writes | 137,656 | $1.25/MTok | $0.17 |
| Uncached input | 555,350 | $1.00/MTok | $0.56 |
| Output | 93,749 | $5.00/MTok | $0.47 |
| **Total** | | | **$1.95** |

## Extrapolation

- **Per chunk**: ~$0.0018
- **Per million chars of source text**: ~$1.80
- Without prompt caching, cost would have been ~$8.20 (cache saved ~76%)
- 3 documents that missed caching accounted for ~$1.05 (54% of total cost for only 15% of chunks)

## Notes

- Prompt caching requires minimum 4,096 tokens for Haiku 4.5. Documents under ~16,000 chars don't benefit. Switching to Sonnet (1,024-token minimum) would cover all documents but at 3x per-token cost.
- All 1,095 prefixes saved to `contextual_cache/` — future runs with same config will skip API calls entirely.
- LLM judge was initialized but not used (chunk-only mode).
