import xml.etree.ElementTree as ET

from fotosort.xmp import render, sidecar_path, write_sidecar

NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xmp": "http://ns.adobe.com/xap/1.0/",
      "dc": "http://purl.org/dc/elements/1.1/"}


def test_render_is_valid_xmp_with_rating_keywords_and_description():
    xml = render(5, ["FotoSort", "FotoSort|zebra", "a & b <c>"], "Sharp eye & good light", label="Green")
    body = xml.split("?>", 1)[1].rsplit("<?xpacket", 1)[0]
    root = ET.fromstring(body)
    desc = root.find(".//rdf:Description", NS)
    assert desc.get(f"{{{NS['xmp']}}}Rating") == "5" and desc.get(f"{{{NS['xmp']}}}Label") == "Green"
    keywords = [li.text for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NS)]
    assert keywords == ["FotoSort", "FotoSort|zebra", "a & b <c>"]
    assert root.find(".//dc:description/rdf:Alt/rdf:li", NS).text == "Sharp eye & good light"


def test_write_sidecar_never_clobbers_existing_files(tmp_path):
    photo = tmp_path / "IMG_1.ORF"
    photo.write_bytes(b"raw")
    assert write_sidecar(photo, 5, ["FotoSort"]) is True
    assert sidecar_path(photo).exists()
    sidecar_path(photo).write_text("lightroom develop settings")
    assert write_sidecar(photo, 5, ["FotoSort"]) is False
    assert sidecar_path(photo).read_text() == "lightroom develop settings"
    assert write_sidecar(photo, 5, ["FotoSort"], overwrite=True) is True
