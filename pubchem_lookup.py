import json
import re
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, Optional


# A real SMILES string only ever contains this character set. Any response
# containing characters outside it (HTML tags, spaces, prose from an error
# page) is rejected outright, regardless of what the HTTP status line said.
_SMILES_CHAR_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$%./\\:*]+$")


CACHE_PATH = Path("data") / "pubchem_smiles_cache.json"
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT"
)


def load_cache(path: Path = CACHE_PATH) -> Dict[str, Optional[str]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: Dict[str, Optional[str]], path: Path = CACHE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def fetch_smiles(name: str, timeout: int = 12, retries: int = 2, retry_delay: float = 2.0) -> Optional[str]:
    url = PUBCHEM_URL.format(name=urllib.parse.quote(name))
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-sL", "-m", str(timeout), url],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
        except subprocess.TimeoutExpired:
            time.sleep(retry_delay * (attempt + 1))
            continue
        text = result.stdout.strip()
        if result.returncode == 0 and text and _SMILES_CHAR_RE.match(text):
            return text
        time.sleep(retry_delay * (attempt + 1))
    return None


def resolve_smiles(
    names: Iterable[str],
    cache_path: Path = CACHE_PATH,
    delay: float = 2.5,
    consecutive_fail_limit: int = 8,
    backoff_seconds: float = 120.0,
) -> Dict[str, Optional[str]]:
    cache = load_cache(cache_path)
    names = sorted(set(names))
    resolved = 0
    missing = 0
    consecutive_failures = 0
    for idx, name in enumerate(names, start=1):
        if name in cache:
            continue
        smiles = fetch_smiles(name)
        if smiles:
            cache[name] = smiles
            resolved += 1
            consecutive_failures = 0
        else:
            # Don't cache the failure yet: a run of failures is very likely a
            # rate-limit block, not genuinely unresolvable names, so back off
            # and retry this same name rather than recording a false negative.
            missing += 1
            consecutive_failures += 1
            if consecutive_failures >= consecutive_fail_limit:
                print(
                    f"{consecutive_failures} consecutive failures (likely rate-limited); "
                    f"backing off {backoff_seconds:.0f}s before retrying '{name}'"
                )
                save_cache(cache, cache_path)
                time.sleep(backoff_seconds)
                consecutive_failures = 0
                retry = fetch_smiles(name)
                if retry:
                    cache[name] = retry
                    resolved += 1
                else:
                    cache[name] = None
            else:
                cache[name] = None
        time.sleep(delay)
        if idx % 20 == 0 or idx == len(names):
            print(f"resolved {idx}/{len(names)} names (this run: {resolved} found, {missing} missing)")
            save_cache(cache, cache_path)
    save_cache(cache, cache_path)
    return cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resolve drug names to SMILES via PubChem, with caching.")
    parser.add_argument("--names_file", type=str, required=True, help="Text file, one drug name per line.")
    args = parser.parse_args()

    names = [line.strip() for line in Path(args.names_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = resolve_smiles(names)
    found = sum(1 for name in names if cache.get(name))
    print(f"total names: {len(names)} | resolved: {found} | missing: {len(names) - found}")
