from speechpt.coherence.keypoint_extractor import Keypoint, extract_keypoints


class Slide:
    def __init__(self):
        self.title = "Intro"
        self.bullet_points = ["- goal", "- data"]
        self.text = "Intro 입니다. 목표를 설명하고 데이터 수집을 소개합니다."


def test_extract_keypoints_includes_title_and_bullets():
    slide = Slide()
    kps = extract_keypoints(slide)
    texts = [k.text for k in kps]
    assert "Intro" in texts
    assert "goal" in " ".join(texts)
    assert any(k.source == "textrank" for k in kps)
