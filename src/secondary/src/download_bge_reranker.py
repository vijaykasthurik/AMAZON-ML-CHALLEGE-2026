import os
import sys
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

sys.stdout.reconfigure(encoding='utf-8')

def download_bge_reranker():
    out_dir = r"d:\mlschool\outputs\kaggle\cross_encoder\bge_reranker_model"
    os.makedirs(out_dir, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "dinnypaulnavis"
        req.kernel_slug = "cross-encoder-finetuning"
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        
        for f in resp.files:
            fn = f.file_name
            if fn.startswith("bge_reranker_model/"):
                basename = os.path.basename(fn)
                dest = os.path.join(out_dir, basename)
                
                print(f"Downloading {fn} -> {dest} ...", flush=True)
                with requests.get(f.url, stream=True) as r:
                    r.raise_for_status()
                    total_bytes = 0
                    with open(dest, "wb") as out_f:
                        for chunk in r.iter_content(chunk_size=8*1024*1024):
                            if chunk:
                                out_f.write(chunk)
                                total_bytes += len(chunk)
                                print(f"  downloaded {total_bytes / (1024*1024):.1f} MB...", flush=True)
                print(f"Successfully downloaded {basename} ({os.path.getsize(dest)/(1024*1024):.1f} MB)", flush=True)

if __name__ == "__main__":
    download_bge_reranker()
