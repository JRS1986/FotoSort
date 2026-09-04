"""XMP sidecars: hand the result to Lightroom, Capture One, Bridge or digiKam
without moving a single file.

For every pick (or every analysed photo with --xmp all) a `<stem>.xmp` is
written next to the original with a star rating, keywords and, for judged
picks, the judge's reason as the description. Existing sidecars are never
overwritten unless asked, because Lightroom keeps develop settings in them.
Lightroom reads sidecars for RAW files; for JPEGs it expects embedded metadata,
so there use Capture One / Bridge / digiKam, or embed with exiftool:
    exiftool -tagsfromfile %d%f.xmp -all:all photo.jpg
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="FotoSort">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmp:Rating="{rating}"{label_attr}>
   <dc:subject>
    <rdf:Bag>
{keywords}
    </rdf:Bag>
   </dc:subject>{description}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def sidecar_path(photo: Path) -> Path:
    return photo.with_suffix(".xmp")


def render(rating: int, keywords: list[str], description: str = "", label: str = "") -> str:
    kw = "\n".join(f"     <rdf:li>{escape(k)}</rdf:li>" for k in keywords)
    desc = ""
    if description:
        desc = ("\n   <dc:description>\n    <rdf:Alt>\n     <rdf:li xml:lang=\"x-default\">"
                f"{escape(description)}</rdf:li>\n    </rdf:Alt>\n   </dc:description>")
    label_attr = f'\n    xmp:Label="{escape(label)}"' if label else ""
    return TEMPLATE.format(rating=rating, keywords=kw, description=desc, label_attr=label_attr)


def write_sidecar(photo: Path, rating: int, keywords: list[str], description: str = "",
                  label: str = "", overwrite: bool = False) -> bool:
    """Write `<stem>.xmp` next to `photo`. Returns False if one exists and
    overwrite is not set."""
    dst = sidecar_path(photo)
    if dst.exists() and not overwrite:
        return False
    dst.write_text(render(rating, keywords, description, label), encoding="utf-8")
    return True
