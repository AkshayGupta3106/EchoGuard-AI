"""Static guardrails complement physical-device UI tests; no microphone access."""
import xml.etree.ElementTree as ET

from echoguard.paths import REPO_ROOT


def test_phone_monitoring_not_exposed_or_permissioned():
    manifest = ET.parse(REPO_ROOT / "app/app/src/main/AndroidManifest.xml").getroot()
    android = "{http://schemas.android.com/apk/res/android}"
    permissions = {node.get(android + "name") for node in manifest.findall("uses-permission")}
    assert permissions == {"android.permission.RECORD_AUDIO"}
    assert not manifest.findall("application/service")
    ui = (REPO_ROOT / "app/app/src/main/java/com/echoguard/ui/MainActivity.kt").read_text(encoding="utf-8")
    assert "ENABLE AUTOMATIC MONITORING" not in ui
    assert "DISABLE AUTOMATIC MONITORING" not in ui
    assert "Analyze speech for cloned voices and scam intent." in ui
    assert "No audio, transcript or score ever leaves this device." not in ui
    assert "Phone-call audio is not supported." not in ui
    assert 'listOf("HOME", "MICROPHONE", "HISTORY")' in ui


def test_status_bar_defaults_and_dynamic_theme_contract():
    for folder, expected in (("values", "true"), ("values-night", "false")):
        root = ET.parse(REPO_ROOT / f"app/app/src/main/res/{folder}/themes.xml").getroot()
        items = {item.get("name"): item.text for item in root.findall("style/item")}
        assert items["android:windowLightStatusBar"] == expected
        assert items["android:windowLightNavigationBar"] == expected
    ui = (REPO_ROOT / "app/app/src/main/java/com/echoguard/ui/MainActivity.kt").read_text(encoding="utf-8")
    assert "enableEdgeToEdge(statusBarStyle = style, navigationBarStyle = navigationStyle)" in ui
    assert "{ isDark }" in ui
    assert "darkColorScheme(" in ui and "onPrimary = Color(0xFF121212)" in ui


def test_leaving_activity_stops_manual_capture():
    ui = (REPO_ROOT / "app/app/src/main/java/com/echoguard/ui/MainActivity.kt").read_text(encoding="utf-8")
    assert "override fun onStop()" in ui and "pipelineViewModel.stopDemo()" in ui


def test_full_transcript_is_saved_and_displayed_without_preview_limit():
    root = REPO_ROOT / "app/app/src/main/java/com/echoguard"
    vm = (root / "ui/PipelineViewModel.kt").read_text(encoding="utf-8")
    history = (root / "pipeline/CallHistoryManager.kt").read_text(encoding="utf-8")
    ui = (root / "ui/MainActivity.kt").read_text(encoding="utf-8")
    assert "transcript = demoState.liveTranscript" in vm
    assert 'put("transcript", it)' in history and 'obj.has("transcript")' in history
    dialog = ui.split("fun HistoryTranscriptDialog", 1)[1].split("fun LiveTab", 1)[0]
    assert "log.transcript ?: log.transcriptSnippet" in dialog
    assert "SelectionContainer" in dialog and "verticalScroll" in dialog
    assert ".take(" not in dialog
    assert "VIEW FULL TRANSCRIPT" in ui and "VIEW SAVED SNIPPET" in ui
