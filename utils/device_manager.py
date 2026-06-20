try:
    import torch
except ImportError:
    torch = None
import cv2
import os
import concurrent.futures
import platform
import subprocess

class DeviceManager:
    _instance = None
    _thread_pool = None
    _process_pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._init_device()
        return cls._instance

    def _init_device(self):
        self.device = "cpu"
        self.gpu_type = None
        
        if torch is not None:
            # 1. Check for CUDA
            if torch.cuda.is_available():
                self.device = "cuda"
                self.gpu_type = "NVIDIA CUDA"
            # 2. Check for DirectML
            elif hasattr(torch, 'dml') and torch.dml.is_available():
                self.device = "dml"
                self.gpu_type = "DirectML"
            # 3. Check for MacOS MPS
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                self.gpu_type = "Apple Metal"
        
        # OpenCV GPU support check
        self.cv2_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False

    def get_device(self):
        return self.device

    def get_gpu_info(self):
        return {
            "device": self.device,
            "gpu_type": self.gpu_type,
            "cv2_gpu": self.cv2_gpu,
            "cpu_count": os.cpu_count()
        }

    def get_thread_pool(self, max_workers=None):
        if self._thread_pool is None:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers or (os.cpu_count() * 2)
            )
        return self._thread_pool

    def get_process_pool(self, max_workers=None):
        if self._process_pool is None:
            # Keep process pool workers reasonable to avoid OOM
            workers = max_workers or min(os.cpu_count(), 4)
            self._process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        return self._process_pool

    @staticmethod
    def get_system_stats():
        """Returns current system resource usage with cross-platform support."""
        stats = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "gpu_usage": 0
        }
        try:
            sys_type = platform.system()
            if sys_type == "Windows":
                # CPU
                cpu_cmd = "wmic cpu get loadpercentage"
                cpu_out = subprocess.check_output(cpu_cmd, shell=True).decode()
                lines = [l.strip() for l in cpu_out.splitlines() if l.strip()]
                if len(lines) > 1:
                    stats["cpu_usage"] = int(lines[1])
                
                # Mem
                mem_cmd = "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value"
                mem_out = subprocess.check_output(mem_cmd, shell=True).decode()
                mem_data = dict(line.split("=") for line in mem_out.splitlines() if "=" in line)
                total = int(mem_data["TotalVisibleMemorySize"])
                free = int(mem_data["FreePhysicalMemory"])
                stats["memory_usage"] = int(((total - free) / total) * 100)
                
            elif sys_type == "Linux":
                # CPU (from /proc/loadavg - 1 min load / cpu_count)
                with open("/proc/loadavg", "r") as f:
                    load = float(f.read().split()[0])
                stats["cpu_usage"] = min(100, int((load / os.cpu_count()) * 100))
                
                # Mem
                with open("/proc/meminfo", "r") as f:
                    meminfo = {line.split(":")[0]: line.split(":")[1].strip() for line in f}
                total = int(meminfo["MemTotal"].split()[0])
                free = int(meminfo["MemAvailable"].split()[0]) # MemAvailable is better than MemFree
                stats["memory_usage"] = int(((total - free) / total) * 100)
                
        except Exception:
            pass
        return stats

# Global instance
device_manager = DeviceManager()
