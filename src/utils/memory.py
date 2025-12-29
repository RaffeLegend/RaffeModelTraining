import torch
from contextlib import contextmanager
import time
import json
import csv
from datetime import datetime
from pathlib import Path

class DetailedMemoryMonitor:
    def __init__(self, verbose=True, log_to_file=False, log_dir="./memory_logs"):
        self.history = []
        self.verbose = verbose
        self.start_time = None
        self.last_memory = 0
        
        # 新增：日志文件功能
        self.log_to_file = log_to_file
        if log_to_file:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建日志文件
            self.log_file = self.log_dir / f"memory_{timestamp}.txt"
            self.csv_file = self.log_dir / f"memory_{timestamp}.csv"
            self.summary_file = self.log_dir / f"memory_summary_{timestamp}.json"
            
            # 初始化CSV文件
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'elapsed_seconds', 'tag', 
                    'allocated_gb', 'cached_gb', 'max_allocated_gb',
                    'delta_gb'
                ])
            
            # 记录开始信息
            self._write_log(f"{'='*60}")
            self._write_log(f"Memory Monitoring Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write_log(f"{'='*60}")
    
    def _write_log(self, message, level="INFO"):
        """写入日志文件"""
        if not self.log_to_file:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
    
    def _write_csv(self, record):
        """写入CSV文件"""
        if not self.log_to_file:
            return
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                record['elapsed'],
                record['tag'],
                record['allocated'],
                record['cached'],
                record['max_allocated'],
                record['delta']
            ])
    
    def _record(self, tag):
        """记录当前显存使用"""
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # 计算变化
        delta = allocated - self.last_memory
        self.last_memory = allocated
        
        # 计算时间
        if self.start_time is None:
            self.start_time = time.time()
            elapsed = 0
        else:
            elapsed = time.time() - self.start_time
        
        record = {
            'tag': tag,
            'allocated': allocated,
            'cached': cached,
            'max_allocated': max_allocated,
            'delta': delta,
            'elapsed': elapsed
        }
        self.history.append(record)
        
        # 写入日志文件
        if self.log_to_file:
            self._write_csv(record)
        
        if self.verbose:
            sign = '+' if delta >= 0 else ''
            log_message = (f"[{elapsed:6.2f}s] {tag:30} | "
                          f"当前: {allocated:6.2f}GB ({sign}{delta:+.2f}GB) | "
                          f"缓存: {cached:6.2f}GB | 峰值: {max_allocated:6.2f}GB")
            print(log_message)
            
            # 同时写入日志文件
            if self.log_to_file:
                self._write_log(log_message)
            
            # 警报
            if delta > 0.5:
                warning_msg = f"严重增长警告: {tag} 增长 {delta:.2f}GB!"
                print(f"  ⚠️⚠️⚠️ {warning_msg}")
                if self.log_to_file:
                    self._write_log(warning_msg, "WARNING")
            elif delta > 0.1:
                warning_msg = f"增长警告: {tag} 增长 {delta:.2f}GB"
                print(f"  ⚠️ {warning_msg}")
                if self.log_to_file:
                    self._write_log(warning_msg, "WARNING")
        
        return allocated
    
    @contextmanager
    def step(self, tag):
        """上下文管理器，自动记录开始和结束"""
        start_mem = self._record(f"开始: {tag}")
        if self.log_to_file:
            self._write_log(f"步骤 '{tag}' 开始")
        
        try:
            yield
        finally:
            end_mem = self._record(f"结束: {tag}")
            total_delta = end_mem - start_mem
            if self.verbose and abs(total_delta) > 0.01:
                sign = '+' if total_delta >= 0 else ''
                step_summary = f"  → {tag} 净变化: {sign}{total_delta:+.2f}GB"
                print(step_summary)
                
                if self.log_to_file:
                    self._write_log(step_summary)
    
    def summary(self, save_to_file=True):
        """打印显存使用总结"""
        print("\n" + "="*80)
        print("显存使用详细总结")
        print("="*80)
        
        if not self.history:
            print("没有记录")
            return
        
        # 找出最大尖峰
        max_spike = 0
        max_spike_tag = ""
        for i in range(1, len(self.history)):
            delta = abs(self.history[i]['allocated'] - self.history[i-1]['allocated'])
            if delta > max_spike:
                max_spike = delta
                max_spike_tag = f"{self.history[i-1]['tag']} → {self.history[i]['tag']}"
        
        # 打印总结
        print(f"总记录数: {len(self.history)}")
        print(f"最大显存尖峰: {max_spike:.2f}GB ({max_spike_tag})")
        print(f"峰值显存使用: {self.history[-1]['max_allocated']:.2f}GB")
        print(f"最终显存使用: {self.history[-1]['allocated']:.2f}GB")
        
        # 打印最耗显存的操作
        print("\n最耗显存的操作（前10）:")
        deltas = []
        for i in range(1, len(self.history)):
            delta = self.history[i]['allocated'] - self.history[i-1]['allocated']
            if delta > 0:  # 只关心增长
                deltas.append((delta, f"{self.history[i-1]['tag']} → {self.history[i]['tag']}"))
        
        deltas.sort(reverse=True)
        for delta, tag in deltas[:10]:
            print(f"  {tag:50} | +{delta:6.2f}GB")
        
        # 打印时间线
        print("\n显存使用时间线:")
        for i, record in enumerate(self.history):
            if i == 0 or record['delta'] != 0:
                print(f"  [{record['elapsed']:6.2f}s] {record['tag']:30} | "
                      f"{record['allocated']:6.2f}GB (Δ{record['delta']:+.2f}GB)")
        
        # 保存总结到文件
        if save_to_file and self.log_to_file:
            self._save_summary_to_file(max_spike, max_spike_tag)
    
    def _save_summary_to_file(self, max_spike, max_spike_tag):
        """保存总结到JSON文件"""
        summary_data = {
            "monitoring_start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "monitoring_duration_seconds": self.history[-1]['elapsed'] if self.history else 0,
            "total_records": len(self.history),
            "peak_memory_usage_gb": self.history[-1]['max_allocated'] if self.history else 0,
            "final_memory_usage_gb": self.history[-1]['allocated'] if self.history else 0,
            "max_memory_spike_gb": max_spike,
            "max_spike_location": max_spike_tag,
            "top_memory_consuming_operations": []
        }
        
        # 收集最耗显存的操作
        deltas = []
        for i in range(1, len(self.history)):
            delta = self.history[i]['allocated'] - self.history[i-1]['allocated']
            if delta > 0:
                deltas.append((delta, self.history[i-1]['tag'], self.history[i]['tag']))
        
        deltas.sort(reverse=True)
        for delta, from_tag, to_tag in deltas[:10]:
            summary_data["top_memory_consuming_operations"].append({
                "from": from_tag,
                "to": to_tag,
                "delta_gb": delta
            })
        
        # 保存到JSON文件
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细总结已保存到: {self.summary_file}")
        print(f"原始数据已保存到: {self.csv_file}")
        print(f"日志文件: {self.log_file}")
    
    def reset_max_memory(self):
        """重置峰值内存统计"""
        torch.cuda.reset_peak_memory_stats()
        print("已重置峰值内存统计")
        
        if self.log_to_file:
            self._write_log("重置峰值内存统计")
    
    def clear_history(self):
        """清空历史记录"""
        self.history.clear()
        self.last_memory = 0
        print("已清空历史记录")
        
        if self.log_to_file:
            self._write_log("清空历史记录")
    
    def log_custom_message(self, message, level="INFO"):
        """记录自定义消息"""
        if self.verbose:
            print(f"[CUSTOM] {message}")
        
        if self.log_to_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] [{level}] {message}"
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + "\n")
    
    def get_current_stats(self):
        """获取当前内存统计"""
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "max_allocated_gb": max_allocated,
            "timestamp": datetime.now().isoformat()
        }