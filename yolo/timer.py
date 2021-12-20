from datetime import datetime as dt


class DTimer:
    """
        with로 감싼 블럭의 실행 시간을 알려줌.
        사용 예시:
            with DTimer() as dtimer:
                foo()
                bar()
            print(dtimer.elapsed)
    """

    def __enter__(self):
        self.start = dt.now()
        return self

    def __exit__(self, *args):
        self.end = dt.now()
        self.elapsed = self.end - self.start
