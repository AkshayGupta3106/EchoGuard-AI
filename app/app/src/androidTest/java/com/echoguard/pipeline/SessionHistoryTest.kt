package com.echoguard.pipeline

import android.content.ContextWrapper
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.echoguard.fusion.Action
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

/** Isolated history persistence check; never reads or changes the user's history. */
@RunWith(AndroidJUnit4::class)
class SessionHistoryTest {
    @Test fun microphoneSummarySurvivesHistoryReload() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val folder = File(context.cacheDir, "history-test-${UUID.randomUUID()}").apply { check(mkdir()) }
        val isolated = object : ContextWrapper(context) { override fun getFilesDir(): File = folder }
        try {
            val transcript = "Generated test transcript. नमस्ते बैंक. \n".repeat(150) + "THE FINAL WORDS"
            val log = CallLog(UUID.randomUUID().toString(), 1234L, "Microphone Session 1", 35,
                Action.WARN, transcript.take(100), 0L, transcript)
            val manager = CallHistoryManager(isolated)
            manager.addLog(log)
            assertEquals(listOf(log), manager.history.value)
            val reloaded = CallHistoryManager(isolated)
            assertEquals(listOf(log), reloaded.history.value)
            assertTrue(reloaded.history.value.single().transcript!!.endsWith("THE FINAL WORDS"))
            reloaded.deleteLog(log.id)
            assertTrue(CallHistoryManager(isolated).history.value.isEmpty())
        } finally { File(folder, "call_history.json").delete(); folder.delete() }
    }

    @Test fun previewOnlySnippetIsPreservedWithoutPretendingItIsFullText() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val folder = File(context.cacheDir, "history-test-${UUID.randomUUID()}").apply { check(mkdir()) }
        val isolated = object : ContextWrapper(context) { override fun getFilesDir(): File = folder }
        try {
            File(folder, "call_history.json").writeText("""[{"id":"preview-only","timestamp":1234,"title":"Demo Call 1","riskScorePercent":0,"action":"MONITOR","transcriptSnippet":"Saved preview","bytesSent":0}]""")
            val manager = CallHistoryManager(isolated)
            val saved = manager.history.value.single()
            assertEquals("Saved preview", saved.transcriptSnippet)
            assertNull(saved.transcript)
            manager.addLog(saved.copy(id = "new", timestamp = 1235L, transcript = "New complete transcript"))
            val loaded = CallHistoryManager(isolated).history.value
            assertNull(loaded.single { it.id == "preview-only" }.transcript)
            assertEquals("New complete transcript", loaded.single { it.id == "new" }.transcript)
        } finally { File(folder, "call_history.json").delete(); folder.delete() }
    }
}
