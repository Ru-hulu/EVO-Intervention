"""Download the pinned CLIP boundary encoder on the machine that will use it."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.error import URLError
from urllib.request import urlopen

MODEL_ID = "openai/clip-vit-large-patch14-336"
REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def verify(path, item):
    if not path.exists() or path.stat().st_size != item["size"]:
        return False
    digest = (hashlib.sha256() if "lfs" in item
              else hashlib.sha1(f'blob {item["size"]}\0'.encode()))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == (item["lfs"]["sha256"] if "lfs" in item else item["blobId"])


def metadata(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co", help="可选的 Hugging Face 兼容镜像地址")
    parser.add_argument("--modelscope-fallback", action="store_true",
                        help="允许从 ModelScope 获取相同文件，逐文件按固定 HF revision 校验")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    info = metadata(f"{endpoint}/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true")
    if info["sha"] != REVISION or info["id"] != MODEL_ID:
        raise ValueError("CLIP 元数据不是固定的官方 revision")
    items = {item["rfilename"]: item for item in info["siblings"]}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt = []
    for name in FILES:
        item, target = items[name], output / name
        if verify(target, item):
            receipt.append({"name": name, "reused": True, "identity": item})
            continue
        if target.exists():
            raise ValueError(f"已有文件校验不符，不覆盖: {target}")
        urls = [f"{endpoint}/{MODEL_ID}/resolve/{REVISION}/{name}?download=true"]
        if args.modelscope_fallback:
            urls.insert(0, "https://modelscope.cn/api/v1/models/AI-ModelScope/clip-vit-large-patch14-336"
                        f"/repo?Revision=master&FilePath={quote(name)}")
        errors = []
        for url in urls:
            try:
                part = target.with_suffix(target.suffix + ".part")
                with urlopen(url, timeout=60) as response:
                    with part.open("wb") as dest:
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            dest.write(chunk)
                if not verify(part, item):
                    raise ValueError("与官方固定 revision 的文件大小/摘要不符")
                part.replace(target)
                receipt.append({"name": name, "source": url, "identity": item})
                print(f"Verified {name}: {item['size']} bytes", flush=True)
                break
            except (URLError, TimeoutError, ValueError) as error:
                errors.append(str(error))
        else:
            raise RuntimeError(f"{name} 下载失败: {errors}")
    (output / "download_receipt.json").write_text(json.dumps(
        {"repo": MODEL_ID, "revision": REVISION, "files": receipt}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
