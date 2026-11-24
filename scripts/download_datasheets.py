from pathlib import Path
import requests

DATASHEETS = {
    "bosch_bme280_datasheet.pdf": [
        "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf",
        "https://ae-bst.resource.bosch.com/media/_tech/media/datasheets/BST-BME280-DS002.pdf",
    ],
    "bosch_bme280_handling_soldering.pdf": [
        "https://www.bosch-sensortec.com/media/boschsensortec/downloads/handling_soldering_mounting_instructions/bst-bme280-hs006.pdf",
        "https://ae-bst.resource.bosch.com/media/_tech/media/application_notes/BST-BME280-HS006.pdf",
    ],
}

OUT_DIR = Path("data/datasheets")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}


def download_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, urls in DATASHEETS.items():
        dest = OUT_DIR / filename
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[SKIP] Already exists: {dest}")
            continue

        for url in urls:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=45)
                resp.raise_for_status()
                if resp.content.startswith(b"%PDF-"):
                    dest.write_bytes(resp.content)
                    print(f"[OK] Saved {dest} ({dest.stat().st_size / 1024:,.1f} KB)")
                    break
            except Exception as exc:
                print(f"[WARN] Mirror failed ({url}): {exc}")
        else:
            raise RuntimeError(f"Could not download {filename}.")


if __name__ == "__main__":
    download_all()
