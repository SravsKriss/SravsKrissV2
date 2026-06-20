import os
import json
import hashlib
import shutil

class CacheManager:
    def __init__(self, cache_dir="temp/cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.tracking_dir = os.path.join(cache_dir, "tracking")
        self.preview_dir = os.path.join(cache_dir, "previews")
        
        for d in [self.tracking_dir, self.preview_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def get_video_hash(self, video_path):
        """Generates a stable hash for a video file based on size and mtime."""
        stat = os.stat(video_path)
        hasher = hashlib.md5()
        hasher.update(video_path.encode())
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime).encode())
        return hasher.hexdigest()

    def save_tracking_data(self, video_hash, mode, data):
        cache_file = os.path.join(self.tracking_dir, f"{video_hash}_{mode}.json")
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def load_tracking_data(self, video_hash, mode):
        cache_file = os.path.join(self.tracking_dir, f"{video_hash}_{mode}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def get_preview_path(self, video_hash, params_hash):
        return os.path.join(self.preview_dir, f"{video_hash}_{params_hash}.mp4")

    def clear_cache(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            # Recreate subdirs
            os.makedirs(self.tracking_dir)
            os.makedirs(self.preview_dir)

# Global instance
cache_manager = CacheManager()
