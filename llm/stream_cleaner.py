"""LLM 스트림 정리기.

LLM 이 토막토막 보내는 텍스트에서
  1) <think> ... </think> 구간은 버리고
  2) 중국어(한자)가 섞여 나오면 모아 뒀다가 한국어로 고쳐서 내보낸다.
결과 조각은 feed()/flush() 가 차례로 돌려주고, 전체 번역문은 .text 로 얻는다.
"""
import re

_CJK = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_KOR_OR_LATIN = re.compile(r"[가-힣a-zA-Z]")


class TranslationStreamCleaner:
    def __init__(self, fix_chinese):
        """fix_chinese: async (앞문장, 중국어) -> 한국어"""
        self._fix_chinese = fix_chinese
        self._pieces = []
        self._buffer = ""
        self._cjk_buffer = ""
        self._thinking = False

    @property
    def text(self) -> str:
        """지금까지 내보낸 번역문 전체."""
        return "".join(self._pieces).strip()

    async def feed(self, chunk: str):
        """새 토막을 넣고, 내보낼 수 있는 조각을 돌려준다."""
        self._buffer += chunk

        if not self._skip_thinking():
            return
        if not self._buffer:
            return
        # 태그가 잘려서 오는 중일 수 있으니 다음 토막을 기다림
        if self._buffer.endswith("<") or self._buffer.endswith("<th"):
            return

        if self._cjk_buffer or _CJK.search(self._buffer):
            self._cjk_buffer += self._buffer
            self._buffer = ""
            async for piece in self._drain_cjk():
                yield piece
        else:
            yield self._emit(self._buffer)
            self._buffer = ""

    async def flush(self):
        """스트림이 끝났을 때 남은 중국어 버퍼를 처리한다."""
        if self._cjk_buffer.strip():
            fixed = await self._fix_chinese(self.text, self._cjk_buffer)
            yield self._emit(f"{fixed} ")
        elif self._cjk_buffer:
            yield self._emit(self._cjk_buffer)

    def _skip_thinking(self) -> bool:
        """<think> 구간 안이면 False (아직 내보낼 게 없음)."""
        if "<think>" in self._buffer and not self._thinking:
            self._thinking = True
        if self._thinking:
            if "</think>" not in self._buffer:
                return False
            self._thinking = False
            self._buffer = self._buffer.split("</think>", 1)[-1]
        return True

    async def _drain_cjk(self):
        """중국어 뒤에 한글/영문이 나오면, 중국어 부분만 고쳐서 같이 내보낸다."""
        match = _KOR_OR_LATIN.search(self._cjk_buffer)
        if not match:
            return

        idx = match.start()
        cjk_part, rest = self._cjk_buffer[:idx], self._cjk_buffer[idx:]

        if cjk_part.strip():
            fixed = await self._fix_chinese(self.text, cjk_part)
            yield self._emit(f"{fixed} ")
        yield self._emit(rest)
        self._cjk_buffer = ""

    def _emit(self, piece: str) -> str:
        self._pieces.append(piece)
        return piece
