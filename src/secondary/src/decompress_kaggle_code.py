import base64
import zlib
import re

with open("outputs/kaggle/ber_c12_code/ber-c12-infer.py", "r", encoding="utf-8") as f:
    content = f.read()

m = re.search(r'CODE\s*=\s*"([^"]+)"', content)
if m:
    b64 = m.group(1)
    raw = zlib.decompress(base64.b64decode(b64), 16 + zlib.MAX_WBITS)
    with open("outputs/kaggle/ber_c12_code/decompressed_infer.py", "wb") as out:
        out.write(raw)
    print(f"Successfully decompressed {len(raw)} bytes to outputs/kaggle/ber_c12_code/decompressed_infer.py")
else:
    print("Could not find CODE pattern")
