# Good Science Slides Manifest (v2)

**Generated:** 2025-11-19T11:51:12.163111

## Canonical Source

**Primary Source:** `/Users/michaelryan/Library/Application Support/precursor/slides/GoodScience_deck_v2.md`

This manifest provides SHA256-locked verification of the Good Science slides deliverables, including the canonical Slidev source, exported PDFs, meeting handouts, and all referenced assets.

## Scope

- Slidev source (canonical): GoodScience_deck_v2.md
- Exported PDF presentations
- Meeting handouts and summaries  
- Referenced assets (images, backgrounds, etc.)
- Export package and documentation

## File Inventory

| File | Exists | Size (bytes) | Modified (ISO) | SHA256 |
|------|--------|--------------|----------------|--------|
| `...plication Support/precursor/slides/GoodScience_deck_v2.md` | ✗ | N/A | N/A | `N/A` |
| `...lication Support/precursor/slides/GoodScience_deck_v2.pdf` | ✗ | N/A | N/A | `N/A` |
| `...ocuments/GoodScience/Deliverables/GoodScience_deck_v2.pdf` | ✗ | N/A | N/A | `N/A` |
| `...odScience/Deliverables/meeting_summary_GoodScience_v2.pdf` | ✗ | N/A | N/A | `N/A` |
| `...odScience/Deliverables/GoodScience_Handout_2025-11-19.pdf` | ✗ | N/A | N/A | `N/A` |
| `...ts/GoodScience/Deliverables/handout_GoodScience_final.pdf` | ✗ | N/A | N/A | `N/A` |
| `.../GoodScience/Deliverables/GoodScience_deck_v2_appendix.md` | ✗ | N/A | N/A | `N/A` |
| `...cience/Deliverables/exports/GoodScience_deck_v2/README.md` | ✗ | N/A | N/A | `N/A` |
| `...cience/Deliverables/exports/GoodScience_deck_v2/export.sh` | ✗ | N/A | N/A | `N/A` |
| `...liverables/exports/GoodScience_deck_v2/TROUBLESHOOTING.md` | ✗ | N/A | N/A | `N/A` |

## Summary

- **Total files tracked:** 10
- **Files found:** 0
- **Files missing:** 10
- **Referenced assets:** 0
- **Total size:** 0 bytes

## Missing Files

- `/Users/michaelryan/Library/Application Support/precursor/slides/GoodScience_deck_v2.md`
- `/Users/michaelryan/Library/Application Support/precursor/slides/GoodScience_deck_v2.pdf`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/GoodScience_deck_v2.pdf`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/meeting_summary_GoodScience_v2.pdf`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/GoodScience_Handout_2025-11-19.pdf`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/handout_GoodScience_final.pdf`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/GoodScience_deck_v2_appendix.md`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/exports/GoodScience_deck_v2/README.md`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/exports/GoodScience_deck_v2/export.sh`
- `/Users/michaelryan/Documents/GoodScience/Deliverables/exports/GoodScience_deck_v2/TROUBLESHOOTING.md`

### Suggested Fixes for Missing Files

1. **Verify file paths:** Confirm the canonical source location and deliverables directory
2. **Re-export slides:** If PDFs are missing, regenerate from the Slidev source
3. **Check asset paths:** Ensure referenced images/backgrounds are in the correct relative locations
4. **Update manifest:** Re-run this script after fixing file locations

## Usage

This manifest enables reproducible verification of the Good Science slides package:

1. **Verify integrity:** Check SHA256 hashes match expected values
2. **Detect changes:** Compare timestamps and hashes across versions  
3. **Reproduce exports:** Use canonical source to regenerate deliverables
4. **Asset validation:** Ensure all referenced assets are present

## Regeneration

To update this manifest:

```bash
cd /path/to/background-agents
python generate_manifest.py > docs/slides_manifest.md
```

## Notes

- **Environment:** This manifest was generated in a sandboxed environment where the user's local file paths are not accessible
- **Template Status:** This serves as a template/framework - when run in the actual environment with access to the files, it will populate with real metadata
- Canonical source is treated as authoritative for content
- SHA256 hashes provide cryptographic verification of file integrity
- Missing assets should be resolved before final distribution
- This manifest is version-controlled for historical tracking

## Environment-Specific Instructions

When running in the actual environment with access to the files:

1. **Verify paths exist:** Ensure the canonical source path is correct
2. **Run the generator:** Execute `python generate_manifest.py > docs/slides_manifest.md`
3. **Review results:** Check that files are found and hashes are generated
4. **Commit updates:** Version control the updated manifest with real data

