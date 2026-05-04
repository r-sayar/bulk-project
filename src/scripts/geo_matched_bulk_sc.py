"""Find GEO studies with matched bulk RNA-seq and scRNA-seq samples.

Two key facts shape the design:
  * Matched bulk and sc samples are usually in SIBLING series (GSEs) bundled
    under a SuperSeries, not in the same series.
  * GEO SOFT family files encode that via "!Series_relation = SuperSeries of:
    GSEnnn" and "!Series_relation = SubSeries of: GSEnnn" lines.

Pipeline:
  1. Run several esearch queries on the `gds` database to enumerate candidate
     GSE series (ones plausibly involving bulk + sc).
  2. For each candidate, fetch the SOFT family file and extract its
     Series_relation links to build a union-find over related series.
  3. For every cluster, classify every GSM across all member series as
     bulk/sc/other. Keep clusters with at least one bulk AND one sc sample.
  4. Pair bulk with sc samples that share (donor, tissue) — donor-only fallback
     when tissue is unavailable. Emit one row per pair.

Usage:
  NCBI_EMAIL=you@example.com NCBI_API_KEY=... \
      python src/scripts/geo_matched_bulk_sc.py --out matched_bulk_sc.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from Bio import Entrez

SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR = .../bulk-project/src/scripts
# parents[0] = src, parents[1] = bulk-project
REPO_ROOT = SCRIPT_DIR.parents[1]
CACHE_DIR = REPO_ROOT / "data" / "geo_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SC_KEYWORDS = (
    "10x", "chromium", "smart-seq", "smartseq", "smart seq",
    "drop-seq", "dropseq", "cel-seq", "celseq", "bd rhapsody",
    "parse biosciences", "splitseq", "split-seq", "microwell",
    "indrop", "indrops", "singleron", "fluidigm c1", "mars-seq",
    "seq-well", "strt-seq", "sci-rna-seq", "snrna", "scrna",
    "single-cell", "single cell", "single-nucleus", "single nucleus",
    "single nuclei",
)

BULK_NEGATIVE = (
    "single-cell", "single cell", "scrna", "snrna", "10x",
    "chromium", "smart-seq", "drop-seq", "single nucleus",
)

DONOR_KEYS = (
    "donor", "subject", "patient", "individual", "participant",
    "biosample", "case id", "donor id", "subject id", "patient id",
    "animal", "mouse id", "sample id",
)

TISSUE_KEYS = (
    "tissue", "organ", "anatomical site", "source tissue",
    "sample type", "region", "body site", "organ part",
)

CONDITION_KEYS = (
    "condition", "disease", "disease state", "treatment",
    "genotype", "status", "group", "diagnosis", "timepoint",
    "cancer type", "tumor type", "tumour type",
)


def setup_entrez() -> None:
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        sys.exit("Set NCBI_EMAIL (and optionally NCBI_API_KEY) in the environment.")
    Entrez.email = email
    key = os.environ.get("NCBI_API_KEY")
    if key:
        Entrez.api_key = key


# ------------------------------------------------------------------
# Stage 1: enumerate candidate series
# ------------------------------------------------------------------

CANDIDATE_QUERIES = [
    # Explicit "bulk and single-cell" phrasing in title/summary.
    '("bulk RNA-seq"[All Fields] OR "bulk RNA sequencing"[All Fields] OR '
    '"bulk transcriptomic"[All Fields]) '
    'AND ("single cell"[All Fields] OR "single-cell"[All Fields] OR '
    '"scRNA-seq"[All Fields] OR "snRNA-seq"[All Fields]) '
    'AND "Homo sapiens"[Organism] AND "gse"[Filter]',
    # Matched / paired phrasing.
    '("matched bulk"[All Fields] OR "paired bulk"[All Fields] OR '
    '"bulk and single"[All Fields] OR "single cell and bulk"[All Fields]) '
    'AND "Homo sapiens"[Organism] AND "gse"[Filter]',
    # SuperSeries that contain scRNA-seq — relation "SuperSeries of" is not
    # directly indexed, so catch by title/summary keyword.
    '("SuperSeries"[All Fields]) AND ("scRNA"[All Fields] OR '
    '"single cell"[All Fields] OR "single-cell"[All Fields]) '
    'AND "Homo sapiens"[Organism] AND "gse"[Filter]',
    # Single-cell studies — we'll use SOFT Series_relation to find their bulk
    # siblings. Capped to keep runtime reasonable.
    '("scRNA-seq"[All Fields] OR "single-cell RNA"[All Fields]) '
    'AND "Homo sapiens"[Organism] AND "gse"[Filter]',
]


def esearch_all(term: str, retmax: int = 10000) -> list[str]:
    handle = Entrez.esearch(db="gds", term=term, retmax=retmax, usehistory="y")
    res = Entrez.read(handle)
    handle.close()
    return list(res["IdList"])


def esummary_batch(uids: list[str], batch: int = 300) -> Iterable[dict]:
    for i in range(0, len(uids), batch):
        chunk = uids[i : i + batch]
        for attempt in range(3):
            try:
                handle = Entrez.esummary(db="gds", id=",".join(chunk))
                docs = Entrez.read(handle)
                handle.close()
                for d in docs:
                    yield d
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"  esummary retry ({e})", file=sys.stderr)
                time.sleep(2)


def collect_candidates(max_per_query: int = 5000) -> dict[str, dict]:
    all_uids: set[str] = set()
    for q in CANDIDATE_QUERIES:
        print(f"[search] {q[:100]}…", file=sys.stderr)
        uids = esearch_all(q, retmax=max_per_query)
        print(f"  -> {len(uids)} hits", file=sys.stderr)
        all_uids.update(uids)
    print(f"[search] total unique UIDs: {len(all_uids)}", file=sys.stderr)

    gse_summaries: dict[str, dict] = {}
    for doc in esummary_batch(sorted(all_uids)):
        acc = doc.get("Accession", "")
        if not acc.startswith("GSE"):
            continue
        gse_summaries[acc] = {
            "title": doc.get("title", ""),
            "summary": doc.get("summary", ""),
            "gdsType": doc.get("gdsType", ""),
            "n_samples": int(doc.get("n_samples", 0) or 0),
            "PDAT": doc.get("PDAT", ""),
        }
    print(f"[search] {len(gse_summaries)} candidate GSE series", file=sys.stderr)
    return gse_summaries


# ------------------------------------------------------------------
# Stage 2: SOFT fetch + parse
# ------------------------------------------------------------------

def soft_url(gse: str) -> str:
    prefix = gse[:-3] + "nnn" if len(gse) > 6 else "GSEnnn"
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/"
        f"soft/{gse}_family.soft.gz"
    )


MAX_SOFT_BYTES = 50 * 1024 * 1024  # 50 MB compressed cap


def fetch_soft(gse: str) -> str | None:
    cache = CACHE_DIR / f"{gse}_family.soft"
    if cache.exists():
        return cache.read_text(errors="replace")
    miss = CACHE_DIR / f"{gse}.miss"
    if miss.exists():
        return None
    url = soft_url(gse)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geo-matched/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                # Check Content-Length; skip giant atlases.
                clen = r.headers.get("Content-Length")
                if clen and int(clen) > MAX_SOFT_BYTES:
                    print(f"  skip {gse}: {int(clen)/1e6:.0f}MB exceeds cap",
                          file=sys.stderr)
                    miss.write_text("oversize")
                    return None
                # Read up to cap+1 to detect unknown-size oversize.
                raw = r.read(MAX_SOFT_BYTES + 1)
                if len(raw) > MAX_SOFT_BYTES:
                    miss.write_text("oversize")
                    return None
            text = gzip.decompress(raw).decode(errors="replace")
            cache.write_text(text)
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                miss.write_text("")
                return None
            time.sleep(2 + attempt)
        except Exception as e:  # noqa: BLE001
            print(f"  soft fetch retry {gse} ({e})", file=sys.stderr)
            time.sleep(2 + attempt)
    return None


@dataclass
class Sample:
    gsm: str
    gse: str = ""
    title: str = ""
    library_strategy: str = ""
    library_source: str = ""
    platform: str = ""
    data_processing: str = ""
    extract_protocol: str = ""
    characteristics: dict[str, str] = field(default_factory=dict)

    def classify(self) -> str:
        ls = self.library_strategy.lower()
        lsrc = self.library_source.lower()
        title_l = self.title.lower()
        chars_blob = " ".join(
            f"{k}={v}" for k, v in self.characteristics.items()
        ).lower()
        # Strong per-sample signals (title + characteristics only).
        # Protocols are shared across samples in a family file — unreliable.
        strong_blob = " ".join([title_l, chars_blob, lsrc]).lower()

        is_rnaseq = (
            "rna-seq" in ls or "rna seq" in ls
            or "transcriptom" in lsrc
        )
        if not is_rnaseq:
            return "other"

        # Definitive: library_source explicitly single-cell.
        if "single cell" in lsrc or "single-cell" in lsrc:
            return "sc"

        # Strong bulk signals in title/characteristics.
        bulk_tokens = (
            "bulk", "whole tissue", "whole tumor", "whole tumour",
            "tumor bulk", "tumour bulk", "population rna",
        )
        if any(t in strong_blob for t in bulk_tokens):
            return "bulk"
        # Title-suffix bulk indicators: "BC01_Tumor", "P3_Pooled", "S1_Bulk"
        if re.search(r"[_-](tumor|tumour|pooled|bulk|whole|population)(\b|$)",
                     title_l):
            return "bulk"

        # Strong sc signals in title/characteristics.
        sc_strong = (
            "single cell", "single-cell", "scrna", "snrna",
            "single nucleus", "single-nucleus", "single nuclei",
            "10x genomics", "chromium",
        )
        if any(t in strong_blob for t in sc_strong):
            return "sc"

        # Per-cell-looking titles: BC01_02, P01_cell12, etc.
        if re.search(r"(cell|nuc|nuclei)[\s_#-]*\d+", title_l):
            return "sc"

        # Fallback to weak signals in protocol.
        proto_blob = (self.extract_protocol + " " + self.data_processing).lower()
        if any(k in proto_blob for k in SC_KEYWORDS):
            return "sc"
        if any(k in proto_blob for k in BULK_NEGATIVE):
            return "other"

        # Default: RNA-Seq + transcriptomic without sc hints => bulk.
        return "bulk"

    def donor(self) -> str:
        for k, v in self.characteristics.items():
            if any(dk in k.lower() for dk in DONOR_KEYS):
                n = normalize(v)
                if n:
                    return n
        m = re.search(
            r"(donor|patient|subject|participant|case|indiv(?:idual)?)[\s_:#-]*([A-Za-z0-9]+)",
            self.title, flags=re.I,
        )
        if m:
            return normalize(f"{m.group(1).lower()}{m.group(2)}")
        return ""

    def tissue(self) -> str:
        for k, v in self.characteristics.items():
            if any(tk in k.lower() for tk in TISSUE_KEYS):
                n = normalize(v)
                if n:
                    return n
        return ""

    def condition(self) -> str:
        for k, v in self.characteristics.items():
            if any(ck in k.lower() for ck in CONDITION_KEYS):
                n = normalize(v)
                if n:
                    return n
        return ""


def normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    # drop trailing punctuation
    return s.strip(".,;:")


def parse_soft(text: str, gse: str) -> tuple[list[Sample], list[str]]:
    """Return (samples, related_series_accessions)."""
    samples: list[Sample] = []
    related: list[str] = []
    current: Sample | None = None
    in_series_header = False

    for raw in text.splitlines():
        if raw.startswith("^SERIES"):
            in_series_header = True
            if current is not None:
                samples.append(current)
                current = None
            continue
        if raw.startswith("^SAMPLE"):
            in_series_header = False
            if current is not None:
                samples.append(current)
            gsm_acc = raw.split("=", 1)[1].strip()
            current = Sample(gsm=gsm_acc, gse=gse)
            continue
        if raw.startswith("^PLATFORM"):
            in_series_header = False
            if current is not None:
                samples.append(current)
                current = None
            continue

        if in_series_header and raw.startswith("!Series_relation"):
            # e.g. "!Series_relation = SuperSeries of: GSE131908"
            m = re.search(r"GSE\d+", raw)
            if m:
                related.append(m.group(0))
            continue

        if current is None:
            continue
        if not raw.startswith("!Sample_"):
            continue
        try:
            key, val = raw[1:].split("=", 1)
        except ValueError:
            continue
        key = key.strip()
        val = val.strip()
        if key == "Sample_title":
            current.title = val
        elif key == "Sample_library_strategy":
            current.library_strategy = val
        elif key == "Sample_library_source":
            current.library_source = val
        elif key == "Sample_platform_id":
            current.platform = val
        elif key == "Sample_data_processing":
            current.data_processing += " " + val
        elif key.startswith("Sample_extract_protocol"):
            current.extract_protocol += " " + val
        elif key.startswith("Sample_characteristics_ch"):
            if ":" in val:
                k, v = val.split(":", 1)
                current.characteristics[k.strip()] = v.strip()

    if current is not None:
        samples.append(current)
    return samples, related


# ------------------------------------------------------------------
# Stage 3: cluster series + pair samples
# ------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def pair_matches(cluster: list[str], samples_by_gse: dict[str, list[Sample]],
                 strict: bool) -> list[dict]:
    """One row per (donor, tissue) with lists of matched bulk + sc GSMs."""
    all_samples = [s for g in cluster for s in samples_by_gse.get(g, [])]
    sc = [s for s in all_samples if s.classify() == "sc"]
    bulk = [s for s in all_samples if s.classify() == "bulk"]
    if not sc or not bulk:
        return []

    def key(s: Sample, use_tissue: bool) -> tuple[str, str]:
        return (s.donor(), s.tissue() if use_tissue else "")

    # Try strict keying first (donor+tissue); if no overlap, fall back to donor-only.
    def build(use_tissue: bool) -> dict[tuple[str, str], dict]:
        idx: dict[tuple[str, str], dict] = {}
        for s in all_samples:
            d, t = key(s, use_tissue)
            if not d:
                continue
            cls = s.classify()
            if cls not in {"bulk", "sc"}:
                continue
            slot = idx.setdefault((d, t), {"bulk": [], "sc": []})
            slot[cls].append(s)
        return idx

    idx = build(True)
    overlap = any(v["bulk"] and v["sc"] for v in idx.values())
    if not overlap and not strict:
        idx = build(False)

    rows: list[dict] = []
    for (d, t), slot in idx.items():
        if not slot["bulk"] or not slot["sc"]:
            continue
        def first(samples: list[Sample], attr: str) -> str:
            for x in samples:
                v = getattr(x, attr)()
                if v:
                    return v
            return ""

        rows.append({
            "cluster": ",".join(sorted(cluster)),
            "donor": d,
            "tissue": t or first(slot["bulk"] + slot["sc"], "tissue"),
            "condition": first(slot["bulk"] + slot["sc"], "condition"),
            "n_bulk": len(slot["bulk"]),
            "n_sc": len(slot["sc"]),
            "bulk_GSMs": ";".join(x.gsm for x in slot["bulk"]),
            "sc_GSMs": ";".join(x.gsm for x in slot["sc"]),
            "bulk_GSEs": ";".join(sorted({x.gse for x in slot["bulk"]})),
            "sc_GSEs": ";".join(sorted({x.gse for x in slot["sc"]})),
            "bulk_platforms": ";".join(sorted({x.platform for x in slot["bulk"] if x.platform})),
            "sc_platforms": ";".join(sorted({x.platform for x in slot["sc"] if x.platform})),
            "bulk_titles_example": slot["bulk"][0].title,
            "sc_titles_example": slot["sc"][0].title,
        })
    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="matched_bulk_sc.tsv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=3000,
                    help="Skip GSE with more samples than this (cell atlases).")
    ap.add_argument("--strict", action="store_true",
                    help="Require donor+tissue on both sides; otherwise donor-only fallback.")
    ap.add_argument("--follow-relations", action="store_true", default=True,
                    help="Follow Series_relation to pull sibling series not in the search hit list.")
    args = ap.parse_args()

    setup_entrez()
    candidates = collect_candidates()

    accessions = sorted(candidates)
    if args.limit:
        accessions = accessions[: args.limit]

    # Phase 1: fetch SOFT concurrently.
    samples_by_gse: dict[str, list[Sample]] = {}
    relations: dict[str, list[str]] = {}
    uf = UnionFind()

    to_fetch = [gse for gse in accessions
                if candidates[gse].get("n_samples", 0) <= args.max_samples]
    print(f"[fetch] {len(to_fetch)} series (<= {args.max_samples} samples)",
          file=sys.stderr)

    def fetch_and_parse(gse: str) -> tuple[str, list[Sample] | None, list[str]]:
        text = fetch_soft(gse)
        if not text:
            return gse, None, []
        s, r = parse_soft(text, gse)
        return gse, s, r

    extra: set[str] = set()
    done = 0
    workers = int(os.environ.get("GEO_WORKERS", "16"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_and_parse, g) for g in to_fetch]
        for fut in as_completed(futures):
            gse, samples, rels = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"[fetch {done}/{len(to_fetch)}] cached={len(samples_by_gse)}",
                      file=sys.stderr)
            if samples is None:
                continue
            samples_by_gse[gse] = samples
            relations[gse] = rels
            uf.add(gse)
            for r in rels:
                extra.add(r)

    # Phase 1b: fetch sibling series via Series_relation (concurrent).
    new_ones = sorted(extra - set(samples_by_gse))
    print(f"[fetch] +{len(new_ones)} sibling series", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_and_parse, g) for g in new_ones]
        for fut in as_completed(futures):
            gse, samples, rels = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"[fetch sib {done}/{len(new_ones)}]", file=sys.stderr)
            if samples is None:
                continue
            samples_by_gse[gse] = samples
            relations[gse] = rels
            uf.add(gse)
            for r in rels:
                uf.union(gse, r)

    # Union first-degree relations too
    for gse, rels in relations.items():
        for r in rels:
            if r in samples_by_gse:
                uf.union(gse, r)

    # Phase 2: build clusters
    clusters: dict[str, list[str]] = defaultdict(list)
    for gse in samples_by_gse:
        clusters[uf.find(gse)].append(gse)
    print(f"[cluster] {len(samples_by_gse)} series -> {len(clusters)} clusters",
          file=sys.stderr)

    # Phase 3: pair within clusters
    out_path = Path(args.out)
    writer = None
    n_rows = 0
    n_clusters_with_hits = 0

    with out_path.open("w", newline="") as fh:
        for root, members in clusters.items():
            rows = pair_matches(members, samples_by_gse, strict=args.strict)
            if not rows:
                continue
            n_clusters_with_hits += 1
            for row in rows:
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()),
                                            delimiter="\t")
                    writer.writeheader()
                writer.writerow(row)
                n_rows += 1
            fh.flush()

    print(f"\n[done] {n_rows} matched pairs across {n_clusters_with_hits} clusters",
          file=sys.stderr)
    print(f"[done] written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
