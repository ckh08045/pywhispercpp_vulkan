"""
pywhispercpp/model.py  –  fork patch
=====================================
원본 대비 변경 사항:
  1. beam_size silent failure 픽스
     - beam_size 지정 시 자동으로 BEAM_SEARCH 전략 전환
     - params.beam_search.beam_size 로 중첩 구조체 정확히 접근
     - AttributeError / TypeError 를 삼키지 않고 명시적 경고 출력
  2. _set_param() 헬퍼: 파라미터 설정 실패 시 WARNING 로깅 (silent 방지)
  3. best_of 도 동일한 방식으로 명시적 처리
"""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from pywhispercpp import _pywhispercpp as pw

logger = logging.getLogger(__name__)


# ── 샘플링 전략 상수 ─────────────────────────────────────────────
SAMPLING_GREEDY      = 0
SAMPLING_BEAM_SEARCH = 1


def _set_param(params, name: str, value) -> bool:
    """
    whisper_full_params 객체에 파라미터를 안전하게 설정.
    반환값: 성공 여부 (False 면 호출부에서 WARNING 출력)
    """
    try:
        setattr(params, name, value)
        return True
    except AttributeError:
        return False
    except TypeError as e:
        logger.warning(f"[pywhispercpp] 파라미터 타입 오류 '{name}={value}': {e}")
        return False


class WhisperSegment:
    """트랜스크립션 결과 세그먼트."""
    def __init__(self, t0: int, t1: int, text: str):
        self.t0   = t0    # 시작 (centiseconds)
        self.t1   = t1    # 종료 (centiseconds)
        self.text = text

    def __repr__(self):
        return f"[{self.t0/100:.2f}s → {self.t1/100:.2f}s] {self.text}"


class Model:
    """
    whisper.cpp Python 바인딩 래퍼.

    모델을 한 번 로드한 뒤 transcribe()를 반복 호출.
    subprocess 방식과 달리 매 호출마다 모델을 다시 로드하지 않음.

    예시::

        model = Model(
            "models/ggml-large-v3-turbo.bin",
            language="ko",
            n_threads=4,
            beam_size=5,   # 자동으로 BEAM_SEARCH 전략 전환
        )
        segments = model.transcribe("audio.wav")
    """

    def __init__(
        self,
        model: str,
        n_threads: int = 4,
        language: str = "ko",
        beam_size: int = 1,   # >1 이면 자동으로 BEAM_SEARCH 전환
        best_of: int = 5,
        print_realtime: bool = False,
        print_progress: bool = False,
        **kwargs,
    ):
        model_path = str(Path(model).resolve())
        if not Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")

        logger.info(f"모델 로딩 중: {model_path}")
        t0 = time.perf_counter()
        self._ctx = pw.whisper_init_from_file(model_path)
        if self._ctx is None:
            raise RuntimeError(f"모델 로드 실패: {model_path}")
        logger.info(f"모델 로드 완료 ({time.perf_counter() - t0:.2f}s)")

        self._language       = language
        self._n_threads      = n_threads
        self._beam_size      = beam_size
        self._best_of        = best_of
        self._print_realtime = print_realtime
        self._print_progress = print_progress
        self._extra_kwargs   = kwargs

    def _build_params(self, **override):
        """
        whisper_full_params 생성 및 설정.

        ★ beam_size 픽스 핵심 ★
        -----------------------------------------------
        원본 문제:
            strategy = GREEDY (기본값)
            setattr(params, 'beam_size', 5)
            → GREEDY params 에는 beam_size 속성이 없음
            → AttributeError 발생
            → 원본 코드가 except 없이 무시 → 빈 결과 반환

        수정:
            beam_size > 1 이면 strategy = BEAM_SEARCH 로 전환
            params.beam_search.beam_size = N  (중첩 구조체 직접 접근)
        -----------------------------------------------
        """
        beam_size = override.get("beam_size", self._beam_size)
        best_of   = override.get("best_of",   self._best_of)

        # 전략 자동 결정
        strategy = SAMPLING_BEAM_SEARCH if beam_size > 1 else SAMPLING_GREEDY
        params = pw.whisper_full_default_params(strategy)

        # 일반 파라미터 (flat 접근 가능한 것들)
        simple = {
            "n_threads":      override.get("n_threads",      self._n_threads),
            "language":       override.get("language",       self._language),
            "print_realtime": override.get("print_realtime", self._print_realtime),
            "print_progress": override.get("print_progress", self._print_progress),
            "translate":      override.get("translate",      False),
            "single_segment": override.get("single_segment", False),
            "no_context":     override.get("no_context",     False),
        }
        for name, value in simple.items():
            if not _set_param(params, name, value):
                logger.warning(f"[pywhispercpp] 파라미터 설정 실패: '{name}={value}'")

        # ── 중첩 구조체 파라미터 ★ 픽스 ★ ──────────────────────
        if strategy == SAMPLING_BEAM_SEARCH:
            try:
                params.beam_search.beam_size = beam_size
                logger.debug(f"beam_search.beam_size = {beam_size}")
            except AttributeError:
                # 구버전 바인딩 호환: flat 접근 fallback
                if not _set_param(params, "beam_size", beam_size):
                    logger.warning(
                        f"[pywhispercpp] beam_size={beam_size} 설정 실패. "
                        "C 바인딩이 beam_search 구조체를 노출하지 않습니다. "
                        "_pywhispercpp.cpp 패치 필요."
                    )
        else:
            # GREEDY: greedy.best_of
            try:
                params.greedy.best_of = best_of
                logger.debug(f"greedy.best_of = {best_of}")
            except AttributeError:
                _set_param(params, "best_of", best_of)

        # 추가 kwargs
        handled = {
            "beam_size", "best_of", "n_threads", "language",
            "print_realtime", "print_progress", "translate",
            "single_segment", "no_context",
        }
        for name, value in {**self._extra_kwargs, **override}.items():
            if name in handled:
                continue
            if not _set_param(params, name, value):
                logger.warning(f"[pywhispercpp] 알 수 없는 파라미터 무시됨: '{name}={value}'")

        return params

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        new_segment_callback: Optional[Callable] = None,
        **override_params,
    ) -> List[WhisperSegment]:
        """
        오디오 파일 또는 float32 numpy 배열을 트랜스크립션.

        Parameters
        ----------
        audio : str | np.ndarray
            WAV 파일 경로 또는 16kHz mono float32 배열
        new_segment_callback : callable, optional
            세그먼트 생성 시 즉시 호출 (실시간 출력용)
        **override_params :
            이 호출에만 적용할 파라미터 재정의
        """
        params = self._build_params(**override_params)

        if isinstance(audio, str):
            audio_path = str(Path(audio).resolve())
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")
            samples = pw.load_wav_file(audio_path)
        elif isinstance(audio, np.ndarray):
            samples = audio.astype(np.float32)
        else:
            raise TypeError(f"audio 타입 오류: {type(audio)}")

        if len(samples) == 0:
            logger.warning("[pywhispercpp] 오디오 샘플이 비어있음 — 빈 결과 반환")
            return []

        segments: List[WhisperSegment] = []

        def _on_segment(seg_data):
            seg = WhisperSegment(
                t0=pw.whisper_full_get_segment_t0(self._ctx, seg_data),
                t1=pw.whisper_full_get_segment_t1(self._ctx, seg_data),
                text=pw.whisper_full_get_segment_text(self._ctx, seg_data).strip(),
            )
            segments.append(seg)
            if new_segment_callback:
                new_segment_callback(seg)

        ret = pw.whisper_full(self._ctx, params, samples, _on_segment)
        if ret != 0:
            logger.error(f"[pywhispercpp] whisper_full() 반환 코드: {ret}")

        return segments

    def transcribe_text(self, audio: Union[str, np.ndarray], **override_params) -> str:
        """전체 텍스트만 반환하는 편의 메서드."""
        segments = self.transcribe(audio, **override_params)
        return " ".join(seg.text for seg in segments if seg.text)

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx is not None:
            try:
                pw.whisper_free(self._ctx)
            except Exception:
                pass
