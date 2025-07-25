import datetime
import os
from skyfield.api import load, PlanetaryConstants
import numpy as np

class LunarCycleCalculator:
    """
    计算月球上特定经纬度点的月昼和月夜转换日期。
    """
    def __init__(self, longitude: float, latitude: float, output_dirfile: str, output_filename: str):
        """
        初始化月球周期计算器。

        Args:
            longitude (float): 月球上的目标经度 (-180 到 180)。
            latitude (float): 月球上的目标纬度 (-90 到 90)。
            output_dirfile (str): 输出文件和 SPICE 内核的存储目录。
            output_filename (str): 输出文件名。
        """
        print("正在初始化计算器并加载星历表...")
        self.longitude = longitude
        self.latitude = latitude
        self.output_dirfile = os.path.abspath(output_dirfile)
        self.output_filename = output_filename

        # 确保输出目录存在且可写
        os.makedirs(self.output_dirfile, exist_ok=True)
        if not os.access(self.output_dirfile, os.R_OK | os.W_OK | os.X_OK):
            raise PermissionError(f"目录 '{self.output_dirfile}' 不可读/写/执行")

        # 初始化 Skyfield 时间尺度
        self.ts = load.timescale()
        try:
            self.eph = load('de421.bsp')  # Skyfield 自动下载
            self.sun = self.eph['sun']
            self.moon = self.eph['moon']
        except Exception as e:
            print(f"加载星历表文件 'de421.bsp' 失败: {e}")
            raise

        # 加载 PlanetaryConstants 和月球参考框架
        try:
            self.pc = PlanetaryConstants()
            self.pc.read_text(load('moon_080317.tf'))
            self.pc.read_text(load('pck00008.tpc'))
            self.pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))
            self.frame = self.pc.build_frame_named('MOON_ME_DE421')
            print("PlanetaryConstants 加载成功，将使用精确月球参考框架")
        except Exception as e:
            print(f"PlanetaryConstants 加载失败: {e}")
            raise
        print("初始化完成。")

    def get_sun_altitude(self, time_utc: datetime.datetime) -> float:
        """
        获取指定时间，太阳在月球表面某点的高度角（度）。

        Args:
            time_utc (datetime.datetime): UTC时间

        Returns:
            float: 太阳高度角（度）
        """
        t = self.ts.utc(time_utc)
        sun_pos = self.sun.at(t).position.km
        moon_pos = self.moon.at(t).position.km

        try:
            observer = self.moon + self.pc.build_latlon_degrees(self.frame, self.latitude, self.longitude)
            surface_vec = observer.at(t).position.km
            normal_vec = surface_vec - moon_pos
        except Exception as e:
            print(f"PlanetaryConstants 计算失败: {e}")
            raise

        sun_vec = sun_pos - surface_vec
        normal_vec /= np.linalg.norm(normal_vec)
        sun_vec /= np.linalg.norm(sun_vec)
        cos_angle = np.dot(sun_vec, normal_vec)
        altitude = np.rad2deg(np.arcsin(cos_angle))
        return altitude

    def find_next_event(self, start_time: datetime.datetime, find_sunrise: bool) -> datetime.datetime:
        """
        查找下一次日出或日落事件。

        Args:
            start_time (datetime.datetime): 起始时间
            find_sunrise (bool): True表示找日出，False表示找日落

        Returns:
            datetime.datetime or None: 事件时间，若未找到则返回None
        """
        step = datetime.timedelta(minutes=10)
        max_search = datetime.timedelta(days=40)
        time = start_time
        last_alt = self.get_sun_altitude(time)

        while time - start_time < max_search:
            time += step
            alt = self.get_sun_altitude(time)
            if (time - start_time).total_seconds() % 3600 == 0:
                print(f"{time}: alt={alt:.2f} degrees")
            if abs(alt) < 1.0:
                step = datetime.timedelta(minutes=1)
            else:
                step = datetime.timedelta(minutes=10)
            if find_sunrise and last_alt <= 0 and alt > 0:
                frac = -last_alt / (alt - last_alt)
                event_time = time - step + step * frac
                return event_time.replace(microsecond=0)
            elif not find_sunrise and last_alt > 0 and alt <= 0:
                frac = last_alt / (last_alt - alt)
                event_time = time - step + step * frac
                return event_time.replace(microsecond=0)
            last_alt = alt
        return None

    def generate_lunar_schedule(self, start_date: datetime.datetime, end_date: datetime.datetime):
        """
        为指定的月球坐标和时间范围生成昼夜变化时间表。

        Args:
            start_date (datetime): 起始时间
            end_date (datetime): 结束时间
        """
        output_path = os.path.join(self.output_dirfile, self.output_filename)
        if os.path.exists(output_path):
            print(f"输出文件 {output_path} 已存在，正在删除...")
            os.remove(output_path)
        
        events = []
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"月球Apollo月震台站坐标 (纬度: {self.latitude}°, 经度: {self.longitude}°) 的昼夜变化时间表\n")
            f.write(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}\n")
            f.write("-" * 50 + "\n\n")

            initial_altitude = self.get_sun_altitude(start_date)
            print(f"初始时间 {start_date}: 太阳高度角={initial_altitude:.2f} 度")
            is_day = initial_altitude > 0
            print(f"初始状态: {'白天' if is_day else '黑夜'}")

            current_time = start_date
            while current_time <= end_date:
                next_event_time = self.find_next_event(current_time, find_sunrise=not is_day)
                if next_event_time is None or next_event_time > end_date:
                    break
                event_type = "月出 (Sunrise)" if not is_day else "月落 (Sunset)"
                events.append((next_event_time, event_type))
                f.write(f"{next_event_time.strftime('%Y-%m-%d %H:%M')} - 发生 {event_type}\n")
                print(f"找到事件: {next_event_time.strftime('%Y-%m-%d %H:%M')} - {event_type}")
                current_time = next_event_time + datetime.timedelta(minutes=1)
                is_day = not is_day

            if events:
                print(f"\n共找到 {len(events)} 个昼夜变化事件:")
                for event_time, event_type in events:
                    print(f"{event_time.strftime('%Y-%m-%d %H:%M')} - {event_type}")
            else:
                print("\n未找到任何昼夜变化事件")

        print(f"\n处理完成！结果已保存至文件: {output_path}")

if __name__ == '__main__':
    utc_tz = datetime.timezone.utc
    start_date = datetime.datetime(1971, 1, 1, 0, 0, 0, tzinfo=utc_tz)
    end_date = datetime.datetime(1971, 12, 31, 23, 59, 59, tzinfo=utc_tz)
    moon_lat = -3.04
    moon_lon = -23.42
    output_dirfile = "/home/tu/documents"
    output_filename = f"Apollo_DayORNight_{start_date.year}.txt"

    calculator = LunarCycleCalculator(
        longitude=moon_lon,
        latitude=moon_lat,
        output_dirfile=output_dirfile,
        output_filename=output_filename
    )
    calculator.generate_lunar_schedule(start_date, end_date)