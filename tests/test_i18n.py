from inferdoctor.i18n import detect_system_language, t


def test_detect_system_language_from_lang():
    assert detect_system_language({"LANG": "zh_CN.UTF-8"}) == "zh"
    assert detect_system_language({"LANG": "ja_JP.UTF-8"}) == "ja"


def test_language_environment_takes_precedence_over_neutral_lang():
    assert detect_system_language({"LANGUAGE": "zh_CN", "LANG": "C.UTF-8"}) == "zh"


def test_locale_precedence_prefers_explicit_messages():
    env = {"LANG": "en_US.UTF-8", "LC_MESSAGES": "ja_JP.UTF-8"}
    assert detect_system_language(env) == "ja"


def test_unsupported_or_neutral_locale_falls_back_to_english():
    assert detect_system_language({"LANG": "C.UTF-8"}) == "en"
    assert detect_system_language({"LANG": "fr_FR.UTF-8"}) == "en"


def test_auto_translation_uses_detected_environment(monkeypatch):
    monkeypatch.setenv("LANGUAGE", "ja_JP")
    monkeypatch.setenv("LANG", "C.UTF-8")

    assert t("dashboard.title", "auto").startswith("InferDoctor - ローカルAI")
