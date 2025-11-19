#!/usr/bin/env python3
"""
Generate Good Science slides manifest with file metadata collection.
This script creates a comprehensive manifest of all slide-related files.
"""

import os
import hashlib
import datetime
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional

def get_file_metadata(file_path: str) -> Dict[str, any]:
    """Collect metadata for a file."""
    try:
        path = Path(file_path)
        if path.exists():
            stat = path.stat()
            
            # Calculate SHA256
            sha256_hash = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return {
                "exists": True,
                "size": stat.st_size,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "sha256": sha256_hash.hexdigest()
            }
        else:
            return {
                "exists": False,
                "size": None,
                "modified": None,
                "sha256": None
            }
    except Exception as e:
        return {
            "exists": False,
            "size": None,
            "modified": None,
            "sha256": None,
            "error": str(e)
        }

def parse_slidev_assets(markdown_path: str) -> List[str]:
    """Parse Slidev markdown to find referenced assets."""
    assets = []
    if not os.path.exists(markdown_path):
        return assets
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find image references: ![alt](path), ![alt](path){...}
        img_pattern = r'!\[.*?\]\(([^)]+)\)'
        assets.extend(re.findall(img_pattern, content))
        
        # Find background images: background: path
        bg_pattern = r'background:\s*([^\s\n]+)'
        assets.extend(re.findall(bg_pattern, content))
        
        # Find other asset references
        asset_patterns = [
            r'src:\s*([^\s\n]+)',  # src: path
            r'image:\s*([^\s\n]+)',  # image: path
            r'logo:\s*([^\s\n]+)',   # logo: path
        ]
        
        for pattern in asset_patterns:
            assets.extend(re.findall(pattern, content))
        
        # Clean up paths - remove quotes and resolve relative paths
        cleaned_assets = []
        base_dir = os.path.dirname(markdown_path)
        
        for asset in assets:
            asset = asset.strip('\'"')
            if not asset.startswith('http'):  # Skip URLs
                if not os.path.isabs(asset):
                    asset = os.path.join(base_dir, asset)
                cleaned_assets.append(os.path.normpath(asset))
        
        return list(set(cleaned_assets))  # Remove duplicates
    except Exception as e:
        print(f"Error parsing assets from {markdown_path}: {e}")
        return []

def generate_manifest():
    """Generate the slides manifest."""
    
    # Define file paths to check
    canonical_source = "/Users/michaelryan/Library/Application Support/precursor/slides/GoodScience_deck_v2.md"
    
    files_to_check = [
        # Canonical source
        canonical_source,
        
        # Expected deliverables
        "/Users/michaelryan/Library/Application Support/precursor/slides/GoodScience_deck_v2.pdf",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/GoodScience_deck_v2.pdf",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/meeting_summary_GoodScience_v2.pdf",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/GoodScience_Handout_2025-11-19.pdf",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/handout_GoodScience_final.pdf",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/GoodScience_deck_v2_appendix.md",
        
        # Export package
        "/Users/michaelryan/Documents/GoodScience/Deliverables/exports/GoodScience_deck_v2/README.md",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/exports/GoodScience_deck_v2/export.sh",
        "/Users/michaelryan/Documents/GoodScience/Deliverables/exports/GoodScience_deck_v2/TROUBLESHOOTING.md",
    ]
    
    # Parse assets from canonical source if it exists
    referenced_assets = parse_slidev_assets(canonical_source)
    all_files = files_to_check + referenced_assets
    
    # Collect metadata for all files
    file_metadata = {}
    for file_path in all_files:
        file_metadata[file_path] = get_file_metadata(file_path)
    
    # Generate manifest content
    timestamp = datetime.datetime.now().isoformat()
    
    manifest_content = f"""# Good Science Slides Manifest (v2)

**Generated:** {timestamp}

## Canonical Source

**Primary Source:** `{canonical_source}`

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
"""
    
    # Add file entries to table
    exists_count = 0
    missing_files = []
    total_size = 0
    
    for file_path, metadata in file_metadata.items():
        exists = "✓" if metadata["exists"] else "✗"
        size = str(metadata["size"]) if metadata["size"] is not None else "N/A"
        modified = metadata["modified"] if metadata["modified"] else "N/A"
        sha256 = metadata["sha256"][:16] + "..." if metadata["sha256"] else "N/A"
        
        # Truncate long paths for display
        display_path = file_path
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]
        
        manifest_content += f"| `{display_path}` | {exists} | {size} | {modified} | `{sha256}` |\n"
        
        if metadata["exists"]:
            exists_count += 1
            if metadata["size"]:
                total_size += metadata["size"]
        else:
            missing_files.append(file_path)
    
    # Add summary
    total_files = len(file_metadata)
    assets_count = len(referenced_assets)
    missing_count = len(missing_files)
    
    manifest_content += f"""
## Summary

- **Total files tracked:** {total_files}
- **Files found:** {exists_count}
- **Files missing:** {missing_count}
- **Referenced assets:** {assets_count}
- **Total size:** {total_size:,} bytes

"""
    
    if missing_files:
        manifest_content += "## Missing Files\n\n"
        for missing_file in missing_files:
            manifest_content += f"- `{missing_file}`\n"
        
        manifest_content += """
### Suggested Fixes for Missing Files

1. **Verify file paths:** Confirm the canonical source location and deliverables directory
2. **Re-export slides:** If PDFs are missing, regenerate from the Slidev source
3. **Check asset paths:** Ensure referenced images/backgrounds are in the correct relative locations
4. **Update manifest:** Re-run this script after fixing file locations

"""
    
    manifest_content += """## Usage

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

- Canonical source is treated as authoritative for content
- SHA256 hashes provide cryptographic verification of file integrity
- Missing assets should be resolved before final distribution
- This manifest is version-controlled for historical tracking
"""
    
    return manifest_content

if __name__ == "__main__":
    manifest = generate_manifest()
    print(manifest)