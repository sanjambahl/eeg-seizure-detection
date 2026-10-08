"""Download the Bonn EEG dataset from its official home (Universitat Pompeu Fabra).

    python scripts/download_data.py

Files land in data/bonn/<set>/. The data are free for research and education
(no commercial or military use). If you use them, cite:

    Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001).
    Indications of nonlinear deterministic and finite dimensional structures in time
    series of brain electrical activity: Dependence on recording region and brain state.
    Physical Review E, 64, 061907.

If a link stops working, download the five zip files by hand from
https://www.upf.edu/web/ntsa/downloads (look for "The Bonn EEG time series")
and unzip them into data/bonn/.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.upf.edu/documents/229517819/234490509/"
FILES = {
    "Z": "Z.zip/9c4a0084-c0d6-3cf6-fe48-8a8767713e67",
    "O": "O.zip/f324f98f-1ade-e912-b89d-e313ac362b6a",
    "N": "N.zip/d4f08e2d-3b27-1a6a-20fe-96dcf644902b",
    "F": "F.zip/8219dcdd-d184-0474-e0e9-1ccbba43aaee",
    "S": "S.zip/7647d3f7-c6bb-6d72-57f7-8f12972896a6",
}


def main(out_dir="data/bonn"):
    out = Path(out_dir)
    for letter, path in FILES.items():
        target = out / letter
        if target.exists() and len(list(target.rglob("*.[tT][xX][tT]"))) >= 100:
            print(f"Set {letter}: already downloaded")
            continue
        print(f"Set {letter}: downloading ...", end=" ", flush=True)
        req = urllib.request.Request(BASE + path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(target)
        n = len(list(target.rglob("*.[tT][xX][tT]")))
        print(f"{n} files")
        if n < 100:
            sys.exit(f"Set {letter} looks incomplete ({n} files). Try the manual download.")
    print(f"Done. Data is in {out}/")


if __name__ == "__main__":
    main()
