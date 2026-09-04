import os
import glob
import zstandard as zstd

def decompress_zst_files():
    ais_dir = r"D:\SIH26143_OilSpill\data\raw\ais"
    zst_files = glob.glob(os.path.join(ais_dir, "*.zst"))
    
    print(f"[*] Found {len(zst_files)} compressed .zst AIS files in {ais_dir}...")
    
    for zpath in zst_files:
        csv_path = zpath.replace(".zst", "")
        if not os.path.exists(csv_path):
            print(f"[↓] Decompressing {os.path.basename(zpath)} -> {os.path.basename(csv_path)}...")
            try:
                with open(zpath, 'rb') as fh_in:
                    dctx = zstd.ZstdDecompressor()
                    with open(csv_path, 'wb') as fh_out:
                        dctx.copy_stream(fh_in, fh_out)
                sz_mb = os.path.getsize(csv_path) / (1024 * 1024)
                print(f"[✓] Successfully decompressed {os.path.basename(csv_path)} ({sz_mb:.1f} MB)")
            except Exception as e:
                print(f"[!] Error decompressing {zpath}: {e}")
        else:
            sz_mb = os.path.getsize(csv_path) / (1024 * 1024)
            print(f"[i] Already decompressed: {os.path.basename(csv_path)} ({sz_mb:.1f} MB)")

if __name__ == "__main__":
    decompress_zst_files()
