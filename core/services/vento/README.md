# Vento Influencer Pipeline

This directory contains the Vento-only discovery, verification, qualification, provider, and daily-batch implementation. Generic and Velit do not import this package during their normal workflows.

## Enable or disable

```env
ENABLE_VENTO_PIPELINE=true
```

Set the value to `false` to remove every `/api/vento/*` route. The lead dashboard probes `/api/vento/config` and hides the Vento mode when the route is unavailable.

No external API key is required for application startup or CSV/XLSX/JSON ingestion.

When an earlier Vento `.csv` or `.xlsx` export is uploaded in history mode, it is used only for historical deduplication during discovery. The browser displays only newly found leads. After the user selects the desired new A/B leads and clicks Export, the original file is uploaded again and a combined copy is generated containing every unchanged old row plus only the selected new leads. Existing CSV rows and columns are preserved; XLSX formulas, styles, and additional worksheets are retained. The uploaded original is never overwritten.

## Optional provider configuration

Use `.env.example` as the source of supported variable names.

For Vento creator leads, Tavily is used for discovery and Apify can optionally be used for structured Instagram/TikTok profile enrichment:

- Required for web discovery: `TAVILY_API_KEY`
- Optional for verified creator profile enrichment: `APIFY_API_TOKEN`
- Enable/disable Apify enrichment: `VENTO_APIFY_ENRICHMENT_ENABLED=true`
- Configure follower gate: `VENTO_MIN_SOCIAL_FOLLOWERS=50000`
- Reject unknown follower counts: `VENTO_REQUIRE_SOCIAL_FOLLOWERS=true`

If Apify is not configured, Vento still runs using Tavily snippets and public profile metadata, but follower counts may be less complete. Apify results are cached in `data/vento/apify_profile_cache.json` by default to avoid paying twice for the same handle during repeat runs.

The older generic provider adapter remains isolated in `providers.py` for compatibility tests, but Plan 3 creator enrichment uses Tavily first and Apify only after real social handles are found.

`GEMINI_API_KEY` or `GOOGLE_API_KEY` enables optional evidence interpretation and value-proposition generation. Neither service is allowed to invent verification data.

Quality V2 is enabled with `VENTO_QUALITY_V2_ENABLED=true`. It does not multiply the usable target into a larger fixed raw target. It searches product intersections first, canonicalizes profiles, rejects directories/stores/corporate/media entities early, enriches promising creators, and adapts query order using observed usable yield. It reuses one enrichment search for the second platform, public email, follower and location evidence where possible. Cached creator evidence avoids repeated daily searches.

Strict mode requires canonical Instagram or TikTok, an individual creator identity, Vento product fit and acceptable location evidence. Generic UGC is returned separately as Paid UGC instead of being represented as an influencer. Missing email and follower data remains explicitly unknown. The previous Plan 1 Tavily settings remain available for rollback when Quality V2 is disabled.

## Isolated storage

The feature creates these worksheets only when Vento storage is first used:

- `Vento Influencers`
- `Vento Review Queue`
- `Vento Daily Runs`

Worksheet names can be overridden through environment variables.

## Full removal

1. Set `ENABLE_VENTO_PIPELINE=false` and deploy. This is sufficient to disable the feature safely.
2. If physical code removal is needed, delete `core/services/vento/`, `api/routes/vento_leads.py`, and `tests/test_vento_*.py`.
3. Remove the clearly marked Vento route block in `api/main.py`, the `mode == "vento"` branches in `core/services/lead_service.py`, the Vento helper block in `core/services/todo_sheet_store.py`, and the Vento panel/functions in `ui/leadgen.html`.
4. Remove the Vento variables from `.env.example` and `openpyxl` from `requirements-vercel.txt` only if no other feature uses Excel.
5. Google Sheets tabs are intentionally not deleted automatically. Archive or delete them manually after confirming the data is no longer needed.
