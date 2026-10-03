package com.echoguard.ui

import android.content.Intent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.activity.compose.setContent
import androidx.compose.runtime.CompositionLocalProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.echoguard.fusion.Action
import com.echoguard.pipeline.CallLog
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Render generated fixtures only; never modifies or displays the user's saved speech. */
@RunWith(AndroidJUnit4::class)
class HistoryTranscriptUiTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    private fun find(root: AccessibilityNodeInfo, text: String): AccessibilityNodeInfo? {
        if (root.text?.toString()?.contains(text) == true) return root
        for (i in 0 until root.childCount) {
            root.getChild(i)?.let { child -> find(child, text)?.let { return it } }
        }
        return null
    }

    private fun node(text: String): AccessibilityNodeInfo {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (System.nanoTime() < deadline) {
            instrumentation.waitForIdleSync()
            instrumentation.uiAutomation.rootInActiveWindow?.let { root -> find(root, text)?.let { return it } }
            Thread.sleep(20)
        }
        error("Missing generated-fixture UI text: ${text.take(40)}")
    }

    private fun click(text: String) {
        var target = node(text)
        while (!target.isClickable) {
            target = target.parent ?: break
        }
        assertTrue(target.performAction(AccessibilityNodeInfo.ACTION_CLICK))
    }

    private fun scrollable(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        if (root.isScrollable) return root
        for (i in 0 until root.childCount) {
            root.getChild(i)?.let { child -> scrollable(child)?.let { return it } }
        }
        return null
    }

    private fun render(log: CallLog, dark: Boolean, verify: () -> Unit) {
        val context = instrumentation.targetContext
        val activity = instrumentation.startActivitySync(Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
        try {
            instrumentation.runOnMainSync {
                activity.setContent {
                    CompositionLocalProvider(LocalThemeIsDark provides dark) {
                        EchoGuardTheme { HistoryItem(log, onDelete = {}) }
                    }
                }
            }
            verify()
        } finally { instrumentation.runOnMainSync { activity.finish() } }
    }

    @Test fun fullTranscriptCanBeOpenedScrolledAndClosed() {
        val transcript = "Generated English and Hindi transcript. नमस्ते बैंक.\n".repeat(80) + "FINAL GENERATED WORDS"
        val log = CallLog("fixture", 1234L, "Generated session", 0, Action.MONITOR, transcript.take(100), 0L, transcript)
        render(log, true) {
            click("VIEW FULL TRANSCRIPT")
            node("FULL TRANSCRIPT")
            assertEquals(transcript, node("FINAL GENERATED WORDS").text.toString())
            val scrolling = scrollable(instrumentation.uiAutomation.rootInActiveWindow ?: error("Missing dialog window"))
            assertNotNull("Transcript dialog must scroll", scrolling)
            assertTrue(scrolling!!.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD))
            click("CLOSE")
            node("VIEW FULL TRANSCRIPT")
        }
    }

    @Test fun previewOnlyEntryExplainsWhyOnlySnippetIsAvailable() {
        val log = CallLog("preview-only-fixture", 1234L, "Generated preview-only session", 0, Action.MONITOR, "Saved generated preview", 0L)
        render(log, false) {
            click("VIEW SAVED SNIPPET")
            node("Only a preview was saved for this entry. The full transcript is unavailable.")
            assertEquals("Saved generated preview", node("Saved generated preview").text.toString())
            click("CLOSE")
            node("VIEW SAVED SNIPPET")
        }
    }
}
