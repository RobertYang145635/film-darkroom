"""会话/卷分组库（v5 §13）：卷、帧、ROI、候选与 RenderIntent 的业务状态。

这里的状态**不**属于 UI：裁切页、转换页、批处理与测试都读写同一对象。
"""

from .models import (BLOCKING_STATUSES, SCHEMA_VERSION, Candidate, CandidateSet,
                     FrameRecord, FrameStatus, GeometryRegions, ImportSession,
                     RollAnalysis, RollGroup)

__all__ = ["BLOCKING_STATUSES", "SCHEMA_VERSION", "Candidate", "CandidateSet",
           "FrameRecord", "FrameStatus", "GeometryRegions", "ImportSession",
           "RollAnalysis", "RollGroup"]
