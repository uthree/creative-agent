"""Bounded, checksummed downloads of official public pilot corpora."""

import hashlib
import io
import json
import shutil
import struct
import tarfile
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def byte_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise ValueError("server ignored bounded byte range")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise ValueError("server returned wrong byte range")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("incomplete byte range")
    return data


class RemoteZip(io.RawIOBase):
    """Seek only over the central directory; never read an entire remote archive."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size <= 0:
            return b""
        if size > 8 * 1024**2:
            raise ValueError("unexpected large ZIP directory")
        data = byte_range(self.url, self.position, self.position + size - 1)
        self.position += size
        return data


def zip_member(url, info, archive_size):
    if info.flag_bits & 1 or info.compress_size > 16 * 1024**2 or info.file_size > 16 * 1024**2:
        raise ValueError("unexpected ZIP member")
    data = byte_range(
        url,
        info.header_offset,
        min(archive_size - 1, info.header_offset + info.compress_size + 4095),
    )
    header = struct.unpack("<IHHHHHIIIHH", data[:30])
    if header[0] != 0x04034B50 or header[3] != info.compress_type:
        raise ValueError("ZIP local header mismatch")
    start = 30 + header[-2] + header[-1]
    raw = data[start : start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        value = decoder.decompress(raw, info.file_size + 1)
        if not decoder.eof or decoder.unconsumed_tail:
            raise ValueError("ZIP decompression exceeds member size")
    elif info.compress_type == zipfile.ZIP_STORED:
        value = raw
    else:
        raise ValueError("unsupported ZIP compression")
    if len(value) != info.file_size or zlib.crc32(value) != info.CRC:
        raise ValueError("ZIP member CRC/size mismatch")
    return value


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url, path, expected_md5, *, limit=1024**3):
    """Resume only a matching HTTP byte range; publish only after checksum verification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path, "md5") != expected_md5:
            raise ValueError("existing archive checksum mismatch")
        return path
    partial = path.with_suffix(path.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if start and response.status == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise ValueError("download resume range mismatch")
        elif response.status == 200:
            start = 0
        else:
            raise ValueError("unexpected download status")
        total, last = start, start
        with partial.open("ab" if start else "wb") as stream:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise ValueError("download exceeds explicit byte budget")
                stream.write(chunk)
                if total - last >= 64 * 1024**2:
                    print(json.dumps({"download": path.name, "bytes": total}), flush=True)
                    last = total
    if digest(partial, "md5") != expected_md5:
        raise ValueError("download checksum mismatch; partial file retained for inspection")
    partial.rename(path)
    return path


def extract(archive, root, *, limit=3 * 1024**3):
    """Copy regular corpus files without extracting links, devices or traversal paths."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    count = total = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("unsafe corpus archive member")
            if member.isdir():
                continue
            if not member.isfile() or member.size > 128 * 1024**2:
                raise ValueError("unexpected corpus archive member")
            total += member.size
            if total > limit:
                raise ValueError("extracted corpus exceeds explicit budget")
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError("corpus member escapes extraction root")
            path.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if path.exists():
                if path.stat().st_size != member.size:
                    raise ValueError("existing extracted file size mismatch")
                expected = hashlib.sha256(source.read()).hexdigest()
                if digest(path) != expected:
                    raise ValueError("existing extracted file content mismatch")
            else:
                with path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
            count += 1
    return {"files": count, "bytes": total, "archive_sha256": digest(archive)}


def download_ranges(url, path, expected_md5, size, *, workers=8):
    """Use bounded ranges on the official server; verify the complete original archive."""
    path = Path(path)
    if path.exists():
        if digest(path, "md5") != expected_md5:
            raise ValueError("existing archive checksum mismatch")
        return path
    if not 0 < size <= 1024**3 or not 1 <= workers <= 8:
        raise ValueError("parallel download budget exceeded")
    parts = path.parent / (path.name + ".chunks")
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 8 * 1024**2
    prior = path.with_suffix(path.suffix + ".part")
    prior_size = prior.stat().st_size if prior.exists() else 0

    def obtain(start):
        end = min(size, start + chunk_size) - 1
        destination = parts / str(start)
        if destination.exists() and destination.stat().st_size == end - start + 1:
            return destination
        if end < prior_size:
            with prior.open("rb") as stream:
                stream.seek(start)
                raw = stream.read(end - start + 1)
        else:
            raw = byte_range(url, start, end)
        destination.write_bytes(raw)
        return destination

    with ThreadPoolExecutor(max_workers=workers) as executor:
        paths = []
        for i, part in enumerate(executor.map(obtain, range(0, size, chunk_size))):
            paths.append(part)
            if i % 8 == 0:
                print(
                    json.dumps({"download_ranges": path.name, "verified_parts": i + 1}), flush=True
                )
    temporary = path.with_suffix(path.suffix + ".assembled")
    with temporary.open("wb") as output:
        for part in paths:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    if digest(temporary, "md5") != expected_md5:
        raise ValueError("complete parallel archive checksum mismatch")
    temporary.replace(path)
    return path


def fetch_public_corpora(root, cache):
    root, cache = Path(root), Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "downloads.json"
    with urllib.request.urlopen("https://www.openslr.org/resources/31/md5sum.txt", timeout=60) as r:
        checksum_text = r.read(4096).decode()
    checksums = {
        line.split()[1].lstrip("*"): line.split()[0]
        for line in checksum_text.splitlines()
        if line.strip()
    }
    releases = [
        (name, "https://www.openslr.org/resources/31/" + name, checksums[name], "minilibrispeech")
        for name in ("train-clean-5.tar.gz", "dev-clean-2.tar.gz")
    ]
    releases.append(
        (
            "babyslakh_16k.tar.gz",
            "https://zenodo.org/records/4603870/files/babyslakh_16k.tar.gz?download=1",
            "311096dc2bde7d61c97e930edbfc7f78",
            "babyslakh",
        )
    )
    records = []
    for name, url, md5, directory in releases:
        print(json.dumps({"event": "fetch", "archive": name}), flush=True)
        archive = (
            download_ranges(url, cache / name, md5, 882818115)
            if directory == "babyslakh"
            else download(url, cache / name, md5)
        )
        record = {
            "name": name,
            "url": url,
            "md5": md5,
            "license": "CC-BY-4.0",
            **extract(archive, root / directory),
        }
        records.append(record)
        manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "verified", **record}), flush=True)
    return records


def fetch_jsut_subset(root, *, count=1024, seed=20261101):
    """Fetch original JSUT basic5000 members with CRC checks, without its 2.7 GB ZIP."""
    if not 64 <= count <= 1536:
        raise ValueError("JSUT pilot permits 64 to 1536 utterances")
    root = Path(root).resolve()
    url, size = "https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip", 2688749844
    with zipfile.ZipFile(RemoteZip(url, size)) as archive:
        entries = archive.infolist()
    waves = [i for i in entries if "/basic5000/wav/" in i.filename and i.filename.endswith(".wav")]
    waves.sort(key=lambda i: hashlib.sha256(f"{seed}:{i.filename}".encode()).hexdigest())
    selected = waves[:count] + [
        i for i in entries if not i.is_dir() and not i.filename.endswith(".wav")
    ]
    if sum(i.file_size for i in selected) > 768 * 1024**2:
        raise ValueError("JSUT subset exceeds explicit extraction budget")
    root.mkdir(parents=True, exist_ok=True)

    def obtain(info):
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe JSUT member path")
        path = root / relative
        if path.exists():
            raw = path.read_bytes()
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                raise ValueError("existing JSUT file differs from official ZIP")
        else:
            raw = zip_member(url, info, size)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(raw)
            temporary.replace(path)
        return {
            "member": info.filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "crc32": info.CRC,
            "bytes": len(raw),
        }

    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, record in enumerate(executor.map(obtain, selected)):
            records.append(record)
            if index % 64 == 0:
                print(json.dumps({"event": "jsut_members", "verified": index + 1}), flush=True)
    result = {
        "dataset": "JSUT-1.1-basic5000-subset",
        "url": url,
        "archive_bytes": size,
        "selection_seed": seed,
        "audio_usage": "non-commercial research/personal only",
        "license_file": "jsut_ver1.1/LICENCE.txt",
        "members": records,
    }
    (root / "downloads.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
