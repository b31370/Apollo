import os
import pandas as pd
import cupy as cp
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from obspy import read
from typing import List, Dict, Tuple, Optional

class ApolloMseedReader:
    """
    地震数据处理类，用于读取和处理 mseed 文件，生成每 group_size 个文件一组的 DataFrame。
    """
    def __init__(self, mseed_folder: str, base_file: str, log_filename: str):
        """
        初始化 ApolloMseedReader 类。

        参数:
            mseed_folder: mseed 文件夹路径
            base_file: 输出文件基础路径
            log_filename: 日志文件名
        """
        try:
            cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError("CUDA 12.2 initialization failed. Ensure CUDA 12.2 is compatible with CuPy 13.5.1.") from e

        self.mseed_folder_path = mseed_folder
        self.base_file_path = base_file
        self.log_file_path = os.path.join(base_file, log_filename)
        self.station_names = ['S12', 'S15', 'S16']
        self.expected_ids = {f"XA.{s}.01.MHZ" for s in self.station_names}
        self.sampling_rate = None
        self.sampling_delta = None

        if not os.path.exists(base_file):
            os.makedirs(base_file)
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        self.logger = self._setup_logging("ApolloMseedReader", self.log_file_path)
        self.logger.info("===== 开始 ApolloMseedReader 处理 =====")

    def _setup_logging(self, name: str, log_file: str) -> logging.Logger:
        """设置线程安全的日志配置。"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _process_single_trace(self, trace, file: str) -> Tuple[Optional[str], Optional[cp.ndarray]]:
        """处理单个 trace，验证并转换为 CuPy 数组，NaN 替换为 -1。"""
        trace_id = trace.get_id()
        if trace_id not in self.expected_ids:
            self.logger.warning(f"文件 {file} 中忽略非预期 trace: {trace_id}")
            return None, None
        try:
            data = cp.array(trace.data, dtype=cp.float32)
            if cp.isnan(data).any():
                self.logger.warning(f"文件 {file} trace {trace_id} 包含 NaN 值，将替换为 -1")
                data = cp.where(cp.isnan(data), -1.0, data)
            return trace_id, data
        except cp.cuda.memory.OutOfMemoryError as e:
            self.logger.error(f"文件 {file} trace {trace_id} 内存不足: {str(e)}")
            return None, None

    def _process_file(self, file: str) -> Tuple[str, Dict[str, cp.ndarray], Optional[Tuple[float, float]]]:
        """
        处理单个 mseed 文件，处理 trace 数量和合并逻辑。

        返回:
            (文件名, {trace_id: 合并后的数据}, 采样信息或错误信息)
        """
        try:
            file_path = os.path.join(self.mseed_folder_path, file)
            stream = read(file_path)
            sampling_info = None
            trace_data = {}

            station_traces = {s: [] for s in self.station_names}
            for trace in stream:
                trace_id = trace.get_id()
                station = trace_id.split('.')[1] if trace_id in self.expected_ids else None
                if station in self.station_names:
                    trace_id, data = self._process_single_trace(trace, file)
                    if trace_id and data is not None:
                        station_traces[station].append(data)
                        if sampling_info is None:
                            sampling_info = (trace.stats.sampling_rate, trace.stats.delta)

            for station in self.station_names:
                if station_traces[station]:
                    try:
                        merged_data = cp.concatenate(station_traces[station], axis=0)
                        trace_data[f"XA.{station}.01.MHZ"] = merged_data
                    except cp.cuda.memory.OutOfMemoryError as e:
                        self.logger.error(f"文件 {file} 合并台站 {station} 数据时内存不足: {str(e)}")
                        return file, {}, f"文件 {file} 合并失败: {str(e)}"

            if not trace_data:
                return file, {}, f"文件 {file} 无有效 trace 数据"
            return file, trace_data, sampling_info
        except Exception as e:
            self.logger.error(f"处理文件 {file} 出错: {str(e)}")
            return file, {}, f"处理文件 {file} 出错: {str(e)}"

    def _create_group_dataframe(self, group_files: List[str], station: str) -> Optional[pd.DataFrame]:
        """
        为一组文件和特定台站创建 DataFrame，短 trace 补 -1。

        参数:
            group_files: 文件列表
            station: 台站名称 (S12, S15, S16)

        返回:
            DataFrame 或 None（如果无数据）
        """
        trace_id = f"XA.{station}.01.MHZ"
        max_len = 0
        file_data = {}
        errors = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            cp.get_default_memory_pool().free_all_blocks()
            results = list(executor.map(self._process_file, group_files))

        for file, data, info in results:
            if isinstance(info, str):
                errors.append(info)
                continue
            if trace_id in data:
                file_data[file] = data[trace_id]
                max_len = max(max_len, len(data[trace_id]))
            if self.sampling_rate is None and info is not None:
                self.sampling_rate, self.sampling_delta = info

        if len(errors) == len(group_files):
            self.logger.error(f"台站 {station} 组 {group_files[0]} 所有文件处理失败")
            return None

        if not file_data:
            self.logger.warning(f"台站 {station} 组 {group_files[0]} 无数据")
            return None

        try:
            def get_file_number(filename: str) -> int:
                digits = ''.join(filter(str.isdigit, filename))
                return int(digits) if digits else float('inf')
            sorted_files = sorted(file_data.keys(), key=get_file_number)

            merged_np = cp.full((max_len, len(file_data)), -1.0, dtype=cp.float32)
            for i, file in enumerate(sorted_files):
                arr = file_data[file]
                merged_np[:len(arr), i] = arr

            df = pd.DataFrame(cp.asnumpy(merged_np), columns=sorted_files)
            self.logger.info(f"台站 {station} 组 {group_files[0]} DataFrame 形状: {df.shape}")
            return df
        except cp.cuda.memory.OutOfMemoryError as e:
            self.logger.error(f"台站 {station} 组 {group_files[0]} 内存不足: {str(e)}")
            return None
        except ValueError as e:
            self.logger.error(f"文件排序失败，组 {group_files[0]}: {str(e)}")
            return None
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    def process_mseed_groups(self, group_size: int) -> Dict[str, List[pd.DataFrame]]:
        """
        处理所有 mseed 文件，生成每 group_size 个文件一组的 DataFrame。

        参数:
            group_size: 每组文件数量

        返回:
            字典，键为台站名称，值为对应 DataFrame 列表
        """
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError(f"Invalid group_size: {group_size}. Must be a positive integer.")

        files = sorted(
            [f for f in os.listdir(self.mseed_folder_path) if f.endswith('.mseed')],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if ''.join(filter(str.isdigit, x)) else float('inf')
        )
        num_files = len(files)
        self.logger.info(f"找到 {num_files} 个 mseed 文件")

        file_groups = [files[i:i+group_size] for i in range(0, num_files, group_size)]
        self.logger.info(f"生成 {len(file_groups)} 组文件 (group_size={group_size})")

        result_dfs = {station: [] for station in self.station_names}
        for i, group in enumerate(file_groups):
            self.logger.info(f"处理第 {i+1} 组文件，共 {len(group)} 个文件")
            for station in self.station_names:
                df = self._create_group_dataframe(group, station)
                if df is not None:
                    result_dfs[station].append(df)
                else:
                    self.logger.warning(f"台站 {station} 组 {i+1} 无有效 DataFrame")

        for station in self.station_names:
            self.logger.info(f"台站 {station} 生成 {len(result_dfs[station])} 个 DataFrame")
        return result_dfs

    def save_to_ascii(self, df: pd.DataFrame, output_path: str) -> None:
        """
        保存 DataFrame 为 ASCII 文件，包含列名，使用空格分隔。

        参数:
            df: 要保存的 DataFrame
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w') as f:
                # Write column names as the first line
                f.write(' '.join(df.columns) + '\n')
                # Write data rows, space-separated
                np.savetxt(f, df.values, fmt='%.6f', delimiter=' ')
            self.logger.info(f"保存 ASCII 文件: {output_path}")
        except Exception as e:
            self.logger.error(f"保存 ASCII 文件 {output_path} 失败: {str(e)}")