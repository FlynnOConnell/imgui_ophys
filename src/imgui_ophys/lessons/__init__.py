"""Every stage, in pipeline order, then the pages under Other. A new one is a package here with a Lesson subclass, added to LESSONS."""

from imgui_ophys.lessons.dprime import DprimeLesson
from imgui_ophys.lessons.motion import MotionLesson
from imgui_ophys.lessons.nmf import NmfLesson
from imgui_ophys.lessons.svd import SvdLesson
from imgui_ophys.lessons.traces import TracesLesson

LESSONS = (MotionLesson, NmfLesson, TracesLesson, DprimeLesson, SvdLesson)

__all__ = ["LESSONS", "DprimeLesson", "MotionLesson", "NmfLesson", "SvdLesson", "TracesLesson"]
