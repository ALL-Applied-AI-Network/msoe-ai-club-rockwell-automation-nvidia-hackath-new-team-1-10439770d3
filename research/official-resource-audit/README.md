# Official Resource Audit

This directory contains The A-Team's reproducible audit of every unique public resource directly linked by the official Fall 2026 Rockwell Automation + NVIDIA hackathon materials.

## Results

- Resources attempted: 94
- Text pages extracted: 64
- Media downloaded to the external archive: 17
- Thin/binary captures: 12
- HTTP errors: 1
- Request failures: 0

Raw captures are intentionally stored outside Git under `E:\Hackathon data\The-A-Team-Hackathon-Research-Archive` to avoid repository bloat. This repository contains the reports and resource indices needed to trace every claim back to its URL and capture metadata.

## Files

- `COMPREHENSIVE_RESOURCE_EVIDENCE.md` — resource-by-resource status, headings, metadata, and extracted evidence.
- `ACCESS_GAPS_AND_FOLLOW_UP.md` — authentication walls and manual inspection queue.
- `RESOURCE_INDEX.csv` / `.json` — machine-readable provenance, redirects, hashes, content types, and archive paths.
- `STRATEGIC_DONT_MISS_BRIEF.md` — decision-oriented synthesis for execution, judging, data, compute, safety, and team planning.

## Scope and reproducibility

The audit begins with the official MAIC content manifest and all eight connected event articles. It follows every direct external link referenced by those articles once, without recursively spidering entire vendor websites. This keeps the evidence set exhaustive for organizer-curated links while avoiding unrelated documentation.

The crawler records the requested URL, redirect destination, HTTP status, content type, byte length, SHA-256 hash, page title, headings, extracted-text length, source article, and archive path. JavaScript shells and authentication walls remain in the index as gaps instead of being treated as successful extractions.
