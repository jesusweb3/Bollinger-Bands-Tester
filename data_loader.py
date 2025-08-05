import pandas as pd
import numpy as np
import os
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Папка с данными не найдена: {data_folder}")

    def load_instrument_data(self, instrument: str, timeframe: str) -> pd.DataFrame:
        """Загружает данные инструмента для указанного таймфрейма"""
        filename = f"{instrument}_{timeframe}.csv"
        filepath = os.path.join(self.data_folder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        # Загружаем CSV с заголовками (header=0 означает что первая строка - заголовки)
        df = pd.read_csv(filepath, header=0)
        df = DataLoader._prepare_dataframe(df)

        if not DataLoader.validate_ohlc_data(df):
            raise ValueError(f"Невалидные OHLC данные в файле {filename}")

        return df

    def load_btc_data(self, timeframe: str) -> pd.DataFrame:
        """Загружает данные BTC для указанного таймфрейма"""
        return self.load_instrument_data("BTCUSDT", timeframe)

    def load_1m_data_for_deltas(self, instrument: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Загружает 1m данные инструмента и BTC для расчета дельт"""
        # Для BTC инструментов загружаем только BTC данные
        if DataLoader.is_btc_instrument(instrument):
            btc_1m = self.load_instrument_data("BTCUSDT", "1m")
            return btc_1m.copy(), btc_1m

        # Для других инструментов загружаем оба
        try:
            instrument_1m = self.load_instrument_data(instrument, "1m")
        except FileNotFoundError:
            raise FileNotFoundError(f"1m данные для {instrument} не найдены - необходимы для расчета дельт")

        try:
            btc_1m = self.load_instrument_data("BTCUSDT", "1m")
        except FileNotFoundError:
            raise FileNotFoundError("BTC 1m данные не найдены - необходимы для расчета дельт")

        instrument_1m, btc_1m = DataLoader.synchronize_data(instrument_1m, btc_1m)

        # Проверяем качество синхронизации
        original_instrument_bars = len(self.load_instrument_data(instrument, "1m"))
        original_btc_bars = len(self.load_instrument_data("BTCUSDT", "1m"))
        synced_bars = len(instrument_1m)

        data_loss_percent = (1 - synced_bars / min(original_instrument_bars, original_btc_bars)) * 100
        if data_loss_percent > 10:  # Более 10% потерь данных
            print(f"Предупреждение: большая потеря данных при синхронизации: {data_loss_percent:.1f}%")
            print(f"Исходные бары - {instrument}: {original_instrument_bars}, BTC: {original_btc_bars}")
            print(f"Синхронизированные бары: {synced_bars}")

        return instrument_1m, btc_1m

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает DataFrame: конвертирует время, индексирует"""
        if 'open_time' not in df.columns:
            raise ValueError("Колонка 'open_time' не найдена в CSV файле")

        # Конвертируем open_time в datetime
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=required_cols + ['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df[['datetime', 'open', 'high', 'low', 'close']]

        return df

    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> bool:
        """Валидирует OHLC данные"""
        try:
            # Проверяем условия OHLC через numpy массивы
            high_values = df['high'].values
            low_values = df['low'].values
            open_values = df['open'].values
            close_values = df['close'].values

            # Находим max и min для каждой строки
            max_open_close = np.maximum(open_values, close_values)
            min_open_close = np.minimum(open_values, close_values)

            # Проверяем условия
            high_valid = high_values >= max_open_close
            low_valid = low_values <= min_open_close

            # Проверяем все ли строки валидны
            all_high_valid = np.all(high_valid)
            all_low_valid = np.all(low_valid)

            if not (all_high_valid and all_low_valid):
                invalid_high = np.sum(~high_valid)
                invalid_low = np.sum(~low_valid)
                total_invalid = invalid_high + invalid_low
                print(f"Предупреждение: найдено {total_invalid} невалидных OHLC баров")
                return False

            return True

        except Exception as validation_error:
            print(f"Ошибка валидации: {validation_error}")
            return False

    @staticmethod
    def synchronize_data(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Синхронизирует два DataFrame по времени"""
        start_time = max(df1['datetime'].min(), df2['datetime'].min())
        end_time = min(df1['datetime'].max(), df2['datetime'].max())

        df1_sync = df1[(df1['datetime'] >= start_time) & (df1['datetime'] <= end_time)].copy()
        df2_sync = df2[(df2['datetime'] >= start_time) & (df2['datetime'] <= end_time)].copy()

        df1_sync = df1_sync.set_index('datetime')
        df2_sync = df2_sync.set_index('datetime')

        common_times = df1_sync.index.intersection(df2_sync.index)

        df1_result = df1_sync.loc[common_times].reset_index()
        df2_result = df2_sync.loc[common_times].reset_index()

        return df1_result, df2_result

    def get_available_timeframes(self, instrument: str) -> List[str]:
        """Возвращает список доступных таймфреймов для инструмента"""
        timeframes = []
        # Все возможные таймфреймы из ТЗ
        standard_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '1H', '4h', '4H', '1d']

        for tf in standard_timeframes:
            filename = f"{instrument}_{tf}.csv"
            filepath = os.path.join(self.data_folder, filename)
            if os.path.exists(filepath):
                timeframes.append(tf)

        return timeframes

    def get_applicable_deltas_for_timeframe(self, timeframe: str) -> List[str]:
        """Возвращает список дельт, применимых для данного таймфрейма"""
        # Конвертируем таймфрейм в минуты для сравнения
        timeframe_minutes = self._timeframe_to_minutes(timeframe)

        # Все возможные дельты и их периоды в минутах
        all_deltas = {
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

        # Возвращаем только те дельты, которые >= текущего таймфрейма
        applicable_deltas = []
        for delta_name, delta_minutes in all_deltas.items():
            if delta_minutes >= timeframe_minutes:
                applicable_deltas.append(delta_name)

        return applicable_deltas

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """Конвертирует таймфрейм в минуты"""
        timeframe = timeframe.lower()

        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            # Обработка особых случаев
            timeframe_map = {
                '1h': 60, '1H': 60,
                '4h': 240, '4H': 240
            }
            return timeframe_map.get(timeframe, 1)  # По умолчанию 1 минута

    @staticmethod
    def is_btc_instrument(instrument: str) -> bool:
        """Проверяет, является ли инструмент BTC"""
        return instrument.upper() == "BTCUSDT"

    def get_file_info(self, instrument: str, timeframe: str) -> dict:
        """Возвращает информацию о файле с данными"""
        filename = f"{instrument}_{timeframe}.csv"
        filepath = os.path.join(self.data_folder, filename)

        if not os.path.exists(filepath):
            return {"exists": False}

        try:
            # Просто проверяем что файл читается
            pd.read_csv(filepath, header=0, nrows=1)
            file_size = os.path.getsize(filepath)

            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)

            return {
                "exists": True,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "estimated_bars": line_count,
                "columns": ['open_time', 'open', 'high', 'low', 'close']
            }
        except Exception as file_error:
            return {"exists": True, "error": str(file_error)}


if __name__ == "__main__":
    # Тестируем с реальными данными
    loader = DataLoader("data")

    print("Тестирование DataLoader с реальными данными...")

    # Тест 1: Загрузка данных BTC
    try:
        btc_data = loader.load_btc_data("1m")
        print(f"✓ BTC данные загружены: {len(btc_data)} баров")
        print(f"  Колонки: {list(btc_data.columns)}")
        print(f"  Временной диапазон: с {btc_data['datetime'].min()} по {btc_data['datetime'].max()}")
        print(f"  Пример данных:\n{btc_data.head(2)}")
    except Exception as exc:
        print(f"✗ Ошибка загрузки BTC данных: {exc}")

    # Тест 2: Загрузка данных XRP
    try:
        xrp_data = loader.load_instrument_data("XRPUSDT", "1m")
        print(f"✓ XRP данные загружены: {len(xrp_data)} баров")
        print(f"  Пример данных:\n{xrp_data.head(2)}")
    except Exception as exc:
        print(f"✗ Ошибка загрузки XRP данных: {exc}")

    # Тест 3: Загрузка 1m данных для дельт
    try:
        xrp_1m, btc_1m_data = loader.load_1m_data_for_deltas("XRPUSDT")
        print(f"✓ 1m данные для дельт загружены: XRP={len(xrp_1m)}, BTC={len(btc_1m_data)}")
    except Exception as exc:
        print(f"✗ Ошибка загрузки 1m данных для дельт: {exc}")

    # Тест 4: Проверка типа инструмента
    print(f"✓ BTCUSDT это BTC: {DataLoader.is_btc_instrument('BTCUSDT')}")
    print(f"✓ XRPUSDT это BTC: {DataLoader.is_btc_instrument('XRPUSDT')}")

    # Тест 5: Доступные таймфреймы
    available_tf = loader.get_available_timeframes("BTCUSDT")
    print(f"✓ Доступные таймфреймы для BTCUSDT: {available_tf}")

    # Тест 6: Проверка применимых дельт для разных таймфреймов
    print(f"✓ Применимые дельты для 1m: {loader.get_applicable_deltas_for_timeframe('1m')}")
    print(f"✓ Применимые дельты для 15m: {loader.get_applicable_deltas_for_timeframe('15m')}")
    print(f"✓ Применимые дельты для 1h: {loader.get_applicable_deltas_for_timeframe('1h')}")

    # Тест 7: Проверка логики BTC инструмента при загрузке дельт
    try:
        btc_self, btc_btc = loader.load_1m_data_for_deltas("BTCUSDT")
        print(f"✓ Логика BTC дельт: self={len(btc_self)}, btc={len(btc_btc)}")
    except Exception as exc:
        print(f"✗ Ошибка логики BTC дельт: {exc}")

    # Тест 8: Информация о файлах
    file_info = loader.get_file_info("BTCUSDT", "1m")
    print(f"✓ Информация о файле: {file_info}")

    print("✓ Все тесты завершены")