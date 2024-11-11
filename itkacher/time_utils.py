import time

_time_cache = {}

class TimeUtils:
    @staticmethod
    def start(tag: str):
        _time_cache[tag] = time.perf_counter()

    @staticmethod
    def reset(tag: str):
        _time_cache[tag] = time.perf_counter()

    @staticmethod
    def end(tag: str): 
        print(tag, (time.perf_counter() - _time_cache[tag]) * 1000)
        _time_cache[tag] = time.perf_counter()