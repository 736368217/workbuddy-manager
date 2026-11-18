"""实时指标端点：口径、桶对齐、并发算法与上游不可达时的退化。

这里有三个特别容易算错、且错了以后「看起来一切正常」的地方：

  · **`request_logs.ts` 是请求结束的时刻**（`gateway._record()` 里 `log_ts` 与
    `latency` 在同一段代码里算），所以真实占用区间是 `[ts - latency, ts)`。
    按 `[ts, ts + latency)` 画会把每条请求推到它结束**之后**——一条 30 秒的流式
    请求会在图上的「未来」再忙半分钟；
  · **并发是「同时最多几条」**（区间最大重叠），不是「这一格里有多少条请求被
    统计到」。后者会把同一格中先后完成的两条算成 2，哪怕它们从不同时在跑；
  · **上限要按 realm 算**：`max_in_flight` 是国内版单账号上限，
    `max_in_flight_global` 是**国际版**账号的上限（不是全池上限）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server import config, db  # noqa: E402
from server.routers import stats  # noqa: E402
from server.services import wb2api  # noqa: E402


def _log(ends_at: int, latency_ms: int, status: int = 200, model: str = 'm1') -> None:
    """写入一条**已在 ends_at 结束**的请求。

    注意第一个参数是结束时刻（= 库里 `ts` 的含义），不是开始时刻。
    """
    db.execute(
        'INSERT INTO request_logs(ts,key_id,ip,model,mapped_model,status,'
        'prompt_tokens,completion_tokens,latency_ms,first_token_ms,ua,error,stream,realm) '
        'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (ends_at, None, '1.2.3.4', model, model, status, 10, 10,
         latency_ms, None, 'UA', None, 1, 'cn'),
    )


def _run(coro):
    return asyncio.run(coro)


class _DbCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp.name) / 't.db'
        db._conn = None
        db.connect()
        stats._PEAK_CACHE.clear()

    def tearDown(self) -> None:
        try:
            db._conn.close()  # type: ignore[union-attr]
        finally:
            db._conn = None
            config.DB_PATH = self._orig_db
            stats._PEAK_CACHE.clear()
            self._tmp.cleanup()


class SpanTest(unittest.TestCase):
    """区间还原：`ts` 是结束时刻。"""

    def test_span_ends_at_ts(self) -> None:
        rows = [{'ts': 1000, 'latency_ms': 30_000}]
        self.assertEqual(stats._request_spans(rows), [(970.0, 1000.0)])

    def test_overlap_uses_half_open_intervals(self) -> None:
        # 首尾相接（前一个在 100 结束、后一个在 100 开始）只算 1
        self.assertEqual(stats._max_overlap([(90.0, 100.0), (100.0, 110.0)]), 1)
        # 真正重叠算 2
        self.assertEqual(stats._max_overlap([(90.0, 105.0), (100.0, 110.0)]), 2)


class PeaksTest(_DbCase):
    def test_overlapping_requests_peak(self) -> None:
        now = int(time.time())
        _log(now - 30, 20_000)   # [-50s, -30s)
        _log(now - 25, 20_000)   # [-45s, -25s) → 与上一条重叠，峰值 2
        _log(now - 5, 1_000)     # 单独，不抬高峰值
        peaks = stats._peaks(1)
        self.assertEqual(peaks['concurrency'], 2)
        self.assertEqual(peaks['samples'], 3)

    def test_adjacent_requests_are_not_concurrent(self) -> None:
        now = int(time.time())
        _log(now - 30, 10_000)   # [-40s, -30s)
        _log(now - 20, 10_000)   # [-30s, -20s) → 首尾相接，仍只算 1
        self.assertEqual(stats._peaks(1)['concurrency'], 1)

    def test_peak_does_not_count_sequential_requests_in_one_minute(self) -> None:
        """同一分钟里 5 条串行请求：RPM 峰值是 5，但并发峰值只能是 1。"""
        now = int(time.time())
        base = now - (now % 60)          # 当前这一分钟
        for i in range(5):
            _log(base + i * 10 + 1, 1_000)   # 每条 1 秒，间隔 10 秒
        peaks = stats._peaks(1)
        self.assertEqual(peaks['rpm'], 5)
        self.assertEqual(peaks['concurrency'], 1, '串行请求不该被算成并发')

    def test_cached_for_60_seconds(self) -> None:
        now = int(time.time())
        _log(now - 10, 100)
        first = stats._peaks(1)
        _log(now - 5, 100)   # 还在缓存窗口内
        self.assertEqual(stats._peaks(1), first)
        stats._PEAK_CACHE[1] = (time.time() - 61, first)
        self.assertEqual(stats._peaks(1)['samples'], 2)


class SeriesTest(_DbCase):
    def test_buckets_align_to_absolute_grid(self) -> None:
        out = stats._series()
        self.assertEqual(out['step'], 15)
        self.assertEqual(out['window'], 180)
        self.assertEqual(out['start'] % 15, 0, '桶边界没对齐到 step 的整数倍')
        self.assertEqual(out['end'] - out['start'], 180)
        self.assertEqual(len(out['rpm']), 12)
        self.assertEqual(len(out['concurrency']), 12)

    def test_request_lands_in_expected_bucket(self) -> None:
        start = stats._series()['start']
        _log(start + 7, 100)            # 在第 0 格结束
        _log(start + 15 * 5 + 3, 100)   # 在第 5 格结束
        out = stats._series()
        self.assertEqual(out['rpm'][0], 1)
        self.assertEqual(out['rpm'][5], 1)
        self.assertEqual(sum(out['rpm']), 2)

    def test_long_request_spans_buckets_backwards(self) -> None:
        """45 秒的请求在 15 秒桶里应当占满 3 格（向前铺，不是向后）。"""
        start = stats._series()['start']
        _log(start + 45, 45_000)        # 占用 [start, start+45) → 第 0/1/2 格
        out = stats._series()
        for i in range(3):
            self.assertGreaterEqual(out['concurrency'][i], 1, f'bucket {i} 没算到并发')
        self.assertEqual(out['concurrency'][3], 0, '请求结束后的桶不该再有并发')
        self.assertEqual(sum(out['rpm']), 1)

    def test_concurrency_is_max_overlap_not_hit_count(self) -> None:
        """同一格内先后完成的两条：命中数 2，但同时只有 1 条 → 必须报 1。"""
        start = stats._series()['start']
        _log(start + 3, 2_000)          # [start+1, start+3)
        _log(start + 6, 2_000)          # [start+4, start+6)
        out = stats._series()
        self.assertEqual(out['rpm'][0], 2)
        self.assertEqual(out['concurrency'][0], 1)

    def test_empty_window_is_all_zero(self) -> None:
        out = stats._series()
        self.assertEqual(set(out['rpm']), {0})
        self.assertEqual(set(out['concurrency']), {0})


class RealtimeEndpointTest(_DbCase):
    def _upstream(self, **overrides) -> dict:
        payload = {
            'connected': True,
            'accounts': [
                {'realm': 'cn', 'in_flight': 1},
                {'realm': 'cn', 'in_flight': 0},
                {'realm': 'cn', 'in_flight': 2},
            ],
            'in_flight_full': 0,
            'healthy': 3,
            'sticky_sessions': 4,
        }
        payload.update(overrides)
        return payload

    def _call(self, payload: dict | None) -> dict:
        status = payload if payload is not None else {'connected': False,
                                                     'error': 'connect failed'}
        with mock.patch.object(wb2api, 'get_status',
                               new=mock.AsyncMock(return_value=status)):
            return _run(stats.realtime(days=1, user={'role': 'admin'}))

    def test_aggregates_inflight_and_capacity(self) -> None:
        cfg = Path(self._tmp.name) / 'config.json'
        cfg.write_text(json.dumps({'pool': {'max_in_flight': 3}}), encoding='utf-8')
        with mock.patch.object(config, 'UPSTREAM_CONFIG', cfg):
            out = self._call(self._upstream())
        self.assertTrue(out['available'])
        self.assertEqual(out['inflight'], 3)       # 1 + 0 + 2
        self.assertEqual(out['busy_accounts'], 2)
        self.assertEqual(out['accounts_total'], 3)
        self.assertEqual(out['per_account_limit'], 3)
        self.assertEqual(out['capacity'], 9)       # 3 个 cn 账号 × 3
        self.assertEqual(out['healthy'], 3)
        self.assertEqual(out['sticky'], 4)
        self.assertNotIn('error', out)

    def test_capacity_uses_global_limit_for_global_realm(self) -> None:
        cfg = Path(self._tmp.name) / 'config.json'
        cfg.write_text(json.dumps({
            'pool': {'max_in_flight': 3, 'max_in_flight_global': 1},
        }), encoding='utf-8')
        payload = self._upstream(accounts=[
            {'realm': 'cn', 'in_flight': 1},
            {'realm': 'cn', 'in_flight': 1},
            {'realm': 'global', 'in_flight': 1},
        ])
        with mock.patch.object(config, 'UPSTREAM_CONFIG', cfg):
            out = self._call(payload)
        self.assertEqual(out['per_account_limit'], 3)
        self.assertEqual(out['per_account_limit_global'], 1)
        self.assertEqual(out['capacity'], 7, '2 个 cn×3 + 1 个 global×1')

    def test_limits_fall_back_to_2(self) -> None:
        with mock.patch.object(config, 'UPSTREAM_CONFIG',
                               Path(self._tmp.name) / 'missing.json'):
            self.assertEqual(stats._per_account_limit(), {'cn': 2, 'global': 2})

    def test_zero_limit_is_normalised_to_2(self) -> None:
        """上游把 0 归一化成 2，这里必须同口径，否则界面会显示「上限 0」。"""
        cfg = Path(self._tmp.name) / 'config.json'
        cfg.write_text(json.dumps({'pool': {'max_in_flight': 0,
                                            'max_in_flight_global': -5}}), encoding='utf-8')
        with mock.patch.object(config, 'UPSTREAM_CONFIG', cfg):
            self.assertEqual(stats._per_account_limit(), {'cn': 2, 'global': 2})

    def test_rpm_windows(self) -> None:
        now = int(time.time())
        for _ in range(3):
            _log(now - 30, 100)     # 60 秒内 3 条
        _log(now - 200, 100)        # 3 分钟内、但在 60 秒外
        out = self._call(self._upstream())
        self.assertEqual(out['rpm_60'], 3)
        self.assertEqual(out['rpm_3m'], 1.0)      # 180 秒窗口：3 / 3
        self.assertEqual(out['rpm_5m'], 0.8)      # 4 / 5

    def test_upstream_down_keeps_local_metrics(self) -> None:
        now = int(time.time())
        _log(now - 10, 100)
        out = self._call(None)
        self.assertFalse(out['available'])
        self.assertIn('error', out)
        self.assertNotIn('capacity', out)
        # 本地口径的数据必须还在：上游挂掉时用户正是靠它判断「刚才有没有断」
        self.assertEqual(out['rpm_60'], 1)
        self.assertGreaterEqual(out['peaks']['samples'], 1)
        self.assertEqual(len(out['series']['concurrency']), 12)

    def test_days_is_clamped(self) -> None:
        self.assertEqual(self._call(self._upstream())['days'], 1)
        with mock.patch.object(wb2api, 'get_status',
                               new=mock.AsyncMock(return_value=self._upstream())):
            self.assertEqual(_run(stats.realtime(days=9999,
                                                 user={'role': 'admin'}))['days'], 90)
            self.assertEqual(_run(stats.realtime(days=0,
                                                 user={'role': 'admin'}))['days'], 1)


if __name__ == '__main__':
    unittest.main()
