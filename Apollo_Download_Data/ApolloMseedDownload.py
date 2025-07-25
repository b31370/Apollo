import os
import logging
import shutil
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from tqdm import tqdm
import glob

from obspy import read


class ApolloMseedDownload:
    """
    ApolloMseedDownload 类用于处理阿波罗任务中的噪声和月震事件数据。
    主要功能包括读取数据文件、筛选时间范围内的噪声数据、转换为日期时间格式、下载波形数据并保存。
    """

    def __init__(self, file_path, base_dir, basic_log_filename, mseed_log_filename):
        """
        初始化 ApolloMseedDownload 类，配置统一的基础路径、输入文件路径、日志文件及数据容器。
        """
        self.file_path = file_path
        self.base_dir = base_dir
        self.output_path = os.path.join(self.base_dir, 'OUTPUT_FILES')
        self.log_dir = os.path.join(self.base_dir, 'logs')

        for dir_path in [self.output_path, self.log_dir]:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"已删除目录及其内容：{dir_path}")
                except Exception as e:
                    print(f"删除目录失败：{dir_path}，原因：{e}")
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        print("已重新创建输出目录和日志目录，程序将全新启动。")

        self.valid_gaps = []
        self.dataframe = None
        self.noise_df = None
        self.moonquake_df = None
        self.datetime_df = None

        self._create_directory(self.log_dir)
        self._create_directory(self.output_path)

        basic_log_file_path = os.path.join(self.log_dir, basic_log_filename)
        if os.path.exists(basic_log_file_path):
            try:
                os.remove(basic_log_file_path)
            except OSError as e:
                raise OSError(f"无法删除基本日志文件 {basic_log_file_path}: {e}")
        self.basic_logger = self._setup_logger('BasicInfoLogger', basic_log_file_path, logging.INFO)

        self.basic_logger.info("===== ApolloMseedDownload 类初始化开始 =====")
        self.basic_logger.info(f"基础目录: {self.base_dir}")
        self.basic_logger.info(f"输出目录: {self.output_path}")
        self.basic_logger.info(f"日志目录: {self.log_dir}")
        self.basic_logger.info(f"输入文件: {self.file_path if self.file_path else '（未指定）'}")

        mseed_log_file_path = os.path.join(self.log_dir, mseed_log_filename)
        if os.path.exists(mseed_log_file_path):
            try:
                os.remove(mseed_log_file_path)
            except OSError as e:
                raise OSError(f"无法删除详细日志文件 {mseed_log_file_path}: {e}")
        self.download_mseed_logger = self._setup_logger('NoiseTimeLogger', mseed_log_file_path, logging.DEBUG)

        self.basic_logger.info("日志系统配置完成。")
        self.basic_logger.info(f"  - 基本日志: {basic_log_file_path}")
        self.basic_logger.info(f"  - 详细日志: {mseed_log_file_path}")
        self.basic_logger.info("===== ApolloMseedDownload 类初始化完成 =====")

    def _create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _setup_logger(self, log_name, log_file, level=logging.INFO):
        logger = logging.getLogger(log_name)
        logger.setLevel(level)
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def download_apollo_noise_mseed(self):
        self.basic_logger.info("调用噪声数据下载接口：download_apollo_noise_mseed")
        download_event = "noise"
        self._download_and_save_seismogram(download_event)

    def download_apollo_moonquake_mseed(self):
        self.basic_logger.info("调用月震事件数据下载接口：download_apollo_moonquake_mseed")
        download_event = "moonquake"
        self._download_and_save_seismogram(download_event)

    def apollo_noise_time(self, start_year, start_day, start_hours, start_minutes, end_year, end_day, end_hours, end_minutes):
        """
        处理指定跨年份和具体时间范围内的噪声数据，生成空余时间段的 DataFrame。
        """
        self.basic_logger.info("===== 开始处理跨年噪声数据筛选 =====")
        self._import_apollo_catalog_to_dataframe(data_type='noise')
        self.basic_logger.info(
            f"目标时间范围: {start_year}年第{start_day}天 {start_hours:02d}:{start_minutes:02d} - "
            f"{end_year}年第{end_day}天 {end_hours:02d}:{end_minutes:02d}"
        )

        time_df = self.dataframe.iloc[:, :6].copy()
        time_df.columns = ['Year', 'Day', 'Start_Hour', 'Start_Minute', 'End_Hour', 'End_Minute']
        self.basic_logger.debug(f"提取的时间数据前5行:\n{time_df.head()}")

        datetime_df = time_df.apply(self._convert_to_datetime, axis=1)
        datetime_df['start_time'] = pd.to_datetime(datetime_df['start_time'])
        datetime_df['end_time'] = pd.to_datetime(datetime_df['end_time'])

        target_start = pd.to_datetime(
            f"{start_year}-{start_day:03d} {start_hours:02d}:{start_minutes:02d}",
            format='%Y-%j %H:%M'
        )
        target_end = pd.to_datetime(
            f"{end_year}-{end_day:03d} {end_hours:02d}:{end_minutes:02d}",
            format='%Y-%j %H:%M'
        )
        self.basic_logger.info(f"筛选区间: {target_start} ~ {target_end}")

        filtered_df = datetime_df[(datetime_df['start_time'] >= target_start) & (datetime_df['end_time'] <= target_end)]
        self.basic_logger.info(f"筛选结果：找到 {len(filtered_df)} 条Apollo噪声事件。")
        if not filtered_df.empty:
            self.basic_logger.debug("噪声事件开始时间-结束时间汇总如下:\n" + filtered_df.to_string())
        else:
            self.basic_logger.warning("本区间未找到任何噪声事件。")

        free_2s = []
        current_time = target_start
        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by='start_time')
            for _, row in sorted_df.iterrows():
                if row['start_time'] > current_time:
                    free_2s.append((current_time, row['start_time']))
                    self.basic_logger.debug(f"新增空余区间: {current_time} ~ {row['start_time']}")
                current_time = max(current_time, row['end_time'])
        if current_time < target_end:
            free_2s.append((current_time, target_end))
            self.basic_logger.debug(f"新增尾部空余区间: {current_time} ~ {target_end}")
        free_df = pd.DataFrame(free_2s, columns=['free_start', 'free_end'])
        self.basic_logger.info(f"背景噪声段数量: {len(free_df)}")
        if not free_df.empty:
            self.basic_logger.debug("背景噪声段开始时间-结束时间汇总如下:\n" + free_df.to_string())
        else:
            self.basic_logger.warning("本区间未检测到可用的背景噪声区间！")

        result_df = free_df.rename(columns={'free_start': 'StartDateTime', 'free_end': 'StopDateTime'})
        self.basic_logger.info(
            f"背景噪声时间段:\n(时间范围: {start_year}年第{start_day}天 {start_hours:02d}:{start_minutes:02d} - "
            f"{end_year}年第{end_day}天 {end_hours:02d}:{end_minutes:02d})\n{result_df.to_string()}"
        )
        self.basic_logger.info("===== 背景噪声时间流数据处理完成 =====")

        self.noise_df = result_df
        return result_df

    def split_noiseDF_intervals_by_two_hours(self, interval):
        """
        将 self.noise_df 中的每个区间按{interval}小时拆分，不足{interval}小时的区间丢弃并详细记录日志。
        """
        if self.noise_df is None or self.noise_df.empty:
            self.basic_logger.warning("split_noise_intervals_by_two_hours: 没有可用的背景噪声区间数据。")
            return pd.DataFrame(columns=["StartDateTime", "StopDateTime"])

        self.basic_logger.info(f"===== 开始拆分背景噪声区间为{interval}小时段 =====")
        result = []
        drop_intervals = []

        for idx, row in self.noise_df.iterrows():
            start = pd.to_datetime(row["StartDateTime"])
            end = pd.to_datetime(row["StopDateTime"])
            total_hours = (end - start).total_seconds() / 3600

            if total_hours < interval:
                drop_intervals.append({
                    "StartDateTime": start,
                    "StopDateTime": end,
                    "Reason": f"区间长度{total_hours:.2f}小时 < {interval}小时，已丢弃"
                })
                continue

            current_start = start
            while (end - current_start).total_seconds() >= interval * 3600:
                current_end = current_start + pd.Timedelta(hours=interval)
                if current_end > end:
                    break
                result.append({"StartDateTime": current_start, "StopDateTime": current_end})
                current_start = current_end

            if (end - current_start).total_seconds() > 0 and (end - current_start).total_seconds() < interval * 3600:
                drop_intervals.append({
                    "StartDateTime": current_start,
                    "StopDateTime": end,
                    "Reason": f"尾部区间仅{(end-current_start).total_seconds()/3600:.2f}小时 < {interval}小时，已丢弃"
                })

        result_df = pd.DataFrame(result)
        nrows = len(result_df)
        self.basic_logger.info(f"{interval}小时区间生成完毕，共 {nrows} 个。丢弃区间 {len(drop_intervals)} 个。")
        if nrows > 0:
            self.basic_logger.info(f"{interval}小时区间详情如下：\n{result_df.to_string(index=False)}")
        else:
            self.basic_logger.info(f"没有生成有效的{interval}小时区间！")

        if drop_intervals:
            self.basic_logger.info(f"===== 以下区间因长度不足{interval}小时已丢弃 =====")
            for d in drop_intervals:
                self.basic_logger.info(f"丢弃区间: {d['StartDateTime']} ~ {d['StopDateTime']}，原因：{d['Reason']}")

        self.basic_logger.info(f"===== {interval}小时区间拆分完成 =====")
        self.basic_logger.info(f"最终共生成{interval}小时区间片段数量：{nrows}")

        self.split_noise_df = result_df
        return result_df
    
    def find_event_by_exact_time(self, year, day, hour, minute):
        """
        根据输入的年、天、小时和分钟，精确查找 self.datetime_df 中的事件，并返回对应的 DataFrame。
        """
        if self.datetime_df is None or self.datetime_df.empty:
            self.basic_logger.warning("self.datetime_df 为空，无法查找事件。")
            return pd.DataFrame(columns=['start_time', 'end_time'])

        # 构造目标时间
        target_date = datetime(year, 1, 1) + timedelta(days=day - 1)
        target_start_time = target_date.replace(hour=hour, minute=minute)

        # 精确查找
        matched = self.datetime_df[self.datetime_df['start_time'] == target_start_time]

        if matched.empty:
            self.basic_logger.info(f"未找到精确匹配的事件: {target_start_time}")
            return pd.DataFrame(columns=['start_time', 'end_time'])

        # 只保留 start_time 和 end_time 两列
        result_df = matched[['start_time', 'end_time']].reset_index(drop=True)
        self.basic_logger.info(f"已找到精确事件: \n{result_df.to_string(index=False)}")
        return result_df

    def filter_by_event_type(self, event_type):
        """
        从 Apollo 目录中筛选指定事件类型的数据，并转换为包含 start_time 和 end_time 的 DataFrame。

        Args:
            event_type (str): 事件类型，选项包括：
                A = classified (matching) deep moonquake
                M = unclassified deep moonquake
                C = meteoroid impact
                H = shallow moonquake
                Z = mostly short-period event
                L = LM impact
                S = S-IVB impact
                X = special type

        Returns:
            pd.DataFrame: 包含 start_time 和 end_time 的 DataFrame，格式与示例一致。
        """
        try:
            self.basic_logger.info(f"===== 开始筛选事件类型为 {event_type} 的 Apollo 数据 =====")
            
            # 确保事件类型有效
            valid_event_types = ['A', 'M', 'C', 'H', 'Z', 'L', 'S', 'X']
            if event_type not in valid_event_types:
                self.basic_logger.error(f"无效的事件类型 {event_type}，必须是 {valid_event_types} 之一")
                raise ValueError(f"事件类型必须是 {valid_event_types} 之一")

            # 调用 _import_apollo_catalog_to_dataframe 获取数据
            self._import_apollo_catalog_to_dataframe(data_type=event_type)
            
            # 筛选指定事件类型
            filtered_df = self.dataframe[self.dataframe['Event_Type'] == event_type]
            self.basic_logger.info(f"找到 {len(filtered_df)} 条事件类型为 {event_type} 的记录")

            if filtered_df.empty:
                self.basic_logger.warning(f"未找到事件类型为 {event_type} 的记录")
                return pd.DataFrame(columns=['start_time', 'end_time'])

            # 提取时间相关列并转换为 datetime
            time_df = filtered_df.iloc[:, :6].copy()
            time_df.columns = ['Year', 'Day', 'Start_Hour', 'Start_Minute', 'End_Hour', 'End_Minute']
            self.basic_logger.debug(f"提取的时间数据前6行:\n{time_df.head()}")

            # 应用 _convert_to_datetime 转换为 datetime 格式
            datetime_df = time_df.apply(self._convert_to_datetime, axis=1)
            datetime_df['start_time'] = pd.to_datetime(datetime_df['start_time'])
            datetime_df['end_time'] = pd.to_datetime(datetime_df['end_time'])

            # 验证时间有效性
            invalid_rows = datetime_df[datetime_df['end_time'] <= datetime_df['start_time']]
            if not invalid_rows.empty:
                self.basic_logger.warning(f"检测到 {len(invalid_rows)} 条无效时间记录（结束时间早于或等于开始时间）：\n{invalid_rows.to_string()}")
                datetime_df = datetime_df[datetime_df['end_time'] > datetime_df['start_time']]
                self.basic_logger.info(f"已过滤无效记录，剩余 {len(datetime_df)} 条有效记录")

            # 存储到类的属性
            self.moonquake_df = datetime_df
            self.basic_logger.info(f"事件类型 {event_type} 的 DataFrame 已生成，包含 {len(datetime_df)} 条记录")
            self.basic_logger.debug(f"事件详情:\n{datetime_df.to_string()}")

            # 新增：输出全部datetime_df到日志
            self.basic_logger.info(f"全部datetime_df数据:\n{datetime_df.to_string()}")

            self.basic_logger.info(f"===== 事件类型 {event_type} 数据处理完成 =====")
            return datetime_df

        except Exception as e:
            self.basic_logger.error(f"筛选事件类型 {event_type} 时出错: {str(e)}")
            raise

    def _download_and_save_seismogram(self,
                                    event="noise",
                                    network='XA',
                                    station='S1*',
                                    channel='MH*',
                                    location='*'):
        """
        根据事件类型一次性完成波形数据的下载和保存
        """
        event_type = event.lower()
        self.basic_logger.info(f"===== 开始批量下载 {event_type} 波形数据 =====")

        if event_type == "noise":
            df = self.noise_df
        elif event_type == "moonquake":
            df = self.moonquake_df
        else:
            self.basic_logger.error("event_type 参数必须是 'noise' 或 'moonquake'")
            raise ValueError("event_type 参数必须是 'noise' 或 'moonquake'")

        if df is None or df.empty:
            self.download_mseed_logger.error(f"数据源为空 (类型: {event_type})")
            self.basic_logger.error(f"数据源为空 (类型: {event_type})，取消下载。")
            return {'saved_files': [], 'failed_items': []}

        mseed_dir = os.path.join(self.output_path, event_type)

        if os.path.isdir(mseed_dir):
            try:
                shutil.rmtree(mseed_dir)
                self.basic_logger.info(f"已删除旧MSEED数据目录: {mseed_dir}")
            except Exception as e:
                self.basic_logger.error(f"删除MSEED数据目录失败 {mseed_dir}: {e}")
                return None

        try:
            os.makedirs(mseed_dir)
            self.basic_logger.info(f"已创建MSEED数据目录: {mseed_dir}")
        except Exception as e:
            self.basic_logger.error(f"创建MSEED数据目录失败 {mseed_dir}: {e}")
            return None

        client = Client("IRIS")
        saved_files = []
        failed_files = []
        count = 1

        total = len(df)

        pbar = tqdm(
            total=total,
            desc=f"Apollo_{event_type} 数据下载与保存",
            unit="窗口",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ascii=">=",
            leave=True
        )

        download_start_time = time.time()

        for idx, row in df.iterrows():
            start = UTCDateTime(row['StartDateTime'])
            end = UTCDateTime(row['StopDateTime'])
            try:
                self.download_mseed_logger.info(f"请求下载: {start} ~ {end} [{network}.{station}.{channel}.{location}]")
                st = client.get_waveforms(network=network,
                                         station=station,
                                         channel=channel,
                                         location=location,
                                         starttime=start,
                                         endtime=end)
            except Exception as e:
                failed_files.append({
                    'start': row['StartDateTime'],
                    'end': row['StopDateTime'],
                    'error': f"下载失败: {e}"
                })
                self.basic_logger.error(f"下载失败 ({idx}): {e}")
                pbar.update(1)
                continue

            try:
                if event_type == 'noise':
                    filename = f"{event_type}_Number_{count:03d}_Stream.mseed"
                    count += 1
                else:
                    filename = f"{event_type}_Stream.mseed"

                filepath = os.path.join(mseed_dir, filename)
                st.write(filepath, format='MSEED')
                saved_files.append(filepath)
                self.download_mseed_logger.info(f"已保存MSEED文件: {filepath}")

                if event_type == 'moonquake':
                    break

            except Exception as e:
                failed_files.append({
                    'start': row['StartDateTime'],
                    'end': row['StopDateTime'],
                    'error': f"保存失败: {e}"
                })
                self.download_mseed_logger.error(f"保存失败 ({idx}): {e}")

            pbar.update(1)

        pbar.close()
        elapsed = time.time() - download_start_time

        print("="*48)
        print(f"下载类型：{event_type}")
        print(f"总窗口数: {total}")
        print(f"成功下载: {len(saved_files)} 个")
        print(f"保存目录: {mseed_dir}")
        print(f"下载耗时: {elapsed:.2f} 秒")
        if failed_files:
            print(f"失败下载: {len(failed_files)} 个")
            print("失败详情：")
            for f in failed_files:
                print(f"  {f['start']} - {f['end']} : {f['error']}")
        else:
            print("全部文件下载并保存成功！")
        print("="*48)

        self.basic_logger.info(
            f"{event_type} 数据处理完成：共 {total} 窗，下载成功 {len(saved_files)}，失败 {len(failed_files)}，耗时 {elapsed:.2f}s"
        )
        if failed_files:
            self.basic_logger.warning("部分下载/保存操作失败，详情如下：")
            for f in failed_files:
                self.basic_logger.warning(f"{f['start']} - {f['end']}: {f['error']}")
        else:
            self.basic_logger.info(f"全部 {event_type} 波形数据下载与保存成功。")

        return {'saved_files': saved_files, 'failed_items': failed_files}

    def _import_apollo_catalog_to_dataframe(self, data_type=None):
        """
        将 Lunar Catalog Nakamura 1981 and updates 1008c 文件转换为 pandas DataFrame。
        """
        try:
            self.basic_logger.info(f"开始导入 Apollo 目录文件到 DataFrame，数据类型: {data_type if data_type else '未指定'}")
            columns = [
                'Year', 'Day', 'Signal_Start_Hour', 'Signal_Start_Minute', 'Signal_End_Hour', 'Signal_End_Minute',
                'Env_11_or_12', 'Env_14', 'Env_15', 'Env_16', 'Availability_11_or_12', 'Availability_14',
                'Availability_15', 'Availability_16', 'Quality_11_or_12', 'Quality_14', 'Quality_15', 'Quality_16',
                'Comments', 'Event_Type', 'Matching_Deep_Moonquake', 'Moonquake_Type', 'Moonquake_Number'
            ]
            data = []

            self.basic_logger.info(f"尝试打开 APOLLO 目录文件：{self.file_path}")
            with open(self.file_path, 'r') as af:
                for idx, line in enumerate(af, start=1):
                    values = [
                        line[2:4], line[5:8], line[9:11], line[11:13], line[14:16], line[16:18],
                        line[19:22], line[23:26], line[27:30], line[31:34], line[36],
                        line[37], line[38], line[39], line[41], line[42], line[43],
                        line[44], line[46:76], line[76], line[77:80], line[81], line[82:85]
                    ]
                    row = {col: np.nan if not val.strip() else val.strip() for col, val in zip(columns, values)}
                    data.append(row)
                    if idx % 100 == 0:
                        self.basic_logger.debug(f"已处理 {idx} 行数据...")

            self.basic_logger.info("APOLLO 目录文件读取完成，开始转换为 DataFrame ...")
            self.dataframe = pd.DataFrame(data)
            self.basic_logger.info(f"成功将 Apollo 目录导入 DataFrame，共有 {len(self.dataframe)} 行数据。")

        except Exception as e:
            self.basic_logger.error(f"读取 Apollo 目录文件并导入 DataFrame 时出错：{str(e)}")
            raise

        try:
            self.basic_logger.info("开始处理 Apollo 目录时间信息以生成 datetime DataFrame ...")
            time_dataframe = self.dataframe.iloc[:, :6].copy()
            time_dataframe.columns = ['Year', 'Day', 'Start_Hour', 'Start_Minute', 'End_Hour', 'End_Minute']

            datetime_df = time_dataframe.apply(self._convert_to_datetime, axis=1)
            datetime_df['start_time'] = pd.to_datetime(datetime_df['start_time'])
            datetime_df['end_time'] = pd.to_datetime(datetime_df['end_time'])

            self.basic_logger.info("Apollo 目录时间数据转换完成。")
            self.datetime_df = datetime_df
            return self.datetime_df

        except Exception as e:
            self.basic_logger.error(f"处理 Apollo 目录时间数据时出错：{str(e)}")
            raise

    def _convert_to_datetime(self, row):
        """
        将时间数据转换为 datetime 格式的函数。
        """
        try:
            self.download_mseed_logger.info(f"正在处理行数据: {row.to_dict()}")
            year = 1900 + int(row['Year'])
            day = int(row['Day'])
            start_hour = int(row['Start_Hour'])
            start_minute = int(row['Start_Minute'])

            end_hour_raw = row['End_Hour']
            end_minute_raw = row['End_Minute']

            if (pd.isna(end_hour_raw) or end_hour_raw == '99' or pd.isna(end_minute_raw) or end_minute_raw == '99'):
                self.download_mseed_logger.warning(
                    f"检测到无效的结束时间 ({end_hour_raw}:{end_minute_raw})，将设置为开始时间 + 2小时"
                )
                end_hour = start_hour + 2
                end_minute = start_minute
                self.download_mseed_logger.info(f"调整后的结束时间为 {end_hour}:{end_minute}")
            else:
                end_hour = int(end_hour_raw)
                end_minute = int(end_minute_raw)

            base_date = datetime(year, 1, 1)
            actual_date = base_date + timedelta(days=day - 1)
            start_time = actual_date.replace(hour=start_hour, minute=start_minute)

            if end_hour > 23:
                self.download_mseed_logger.info(f"结束小时 {end_hour} 超过23，将调整时间")
                total_hours = end_hour + (end_minute / 60.0)
                adjusted_days = int(total_hours // 24)
                remaining_hours = int(total_hours % 24)
                remaining_minutes = int((total_hours % 1) * 60)
                end_date = actual_date + timedelta(days=adjusted_days)
                end_time = end_date.replace(hour=remaining_hours, minute=remaining_minutes)
                if end_date.year > actual_date.year:
                    self.download_mseed_logger.info(f"结束时间跨年: {actual_date.year} 到 {end_date.year}")
            else:
                end_time = actual_date.replace(hour=end_hour, minute=end_minute)

            if end_time < start_time:
                self.download_mseed_logger.info("结束时间早于开始时间，自动增加1天")
                end_time += timedelta(days=1)
                if end_time.year > start_time.year:
                    self.download_mseed_logger.info(f"调整后结束时间跨年: {start_time.year} 到 {end_time.year}")

            self.download_mseed_logger.info(f"成功转换 - 开始时间: {start_time}, 结束时间: {end_time}")
            return pd.Series({'start_time': start_time, 'end_time': end_time})

        except (ValueError, KeyError) as e:
            self.download_mseed_logger.error(f"时间转换出错: {str(e)}")
            self.download_mseed_logger.error(f"问题行数据: {row.to_dict()}")
            raise ValueError(f"无法将行转换为 datetime: {str(e)}。行数据: {row.to_dict()}")
        
if __name__ == '__main__':
    base_dir = '/home/tu/code/Apollo_Download_Data'
    file_path = '/home/tu/documents/event/levent.1008c'
    basic_log_filename = 'basic_info.log'
    mseed_log_filename = 'mseed_download.log'

    downloader = ApolloMseedDownload(file_path, base_dir, basic_log_filename, mseed_log_filename)

    meteoroid_df = downloader.filter_by_event_type('C')
    print(meteoroid_df.head())

    moon_start_year = 1976
    moon_start_day = 319
    moon_start_hour = 23
    moon_start_minute = 16
    moon_df = downloader.find_event_by_exact_time(moon_start_year, moon_start_day, moon_start_hour, moon_start_minute)
    print(moon_df.head())

    noise_start_year = 1976
    noise_start_day = 306
    noise_start_hour = 12
    noise_start_minute = 0
    noise_end_year = 1976
    noise_end_day = 319
    noise_end_hour = 23
    noise_end_minute = 13

    # 示例筛选噪声时间段
    noise_df = downloader.apollo_noise_time(
        noise_start_year, noise_start_day, noise_start_hour, noise_start_minute,
        noise_end_year, noise_end_day, noise_end_hour, noise_end_minute
    )
    print(noise_df.head())


    
