package com.echoguard.pipeline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.Executors
import java.util.concurrent.locks.ReentrantReadWriteLock

/** Verify shutdown ordering without opening the microphone or placing a call. */
@RunWith(AndroidJUnit4::class)
class PipelineLifecycleTest {
    @Test fun realPipelineCanScoreGeneratedSilenceAndRestart() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val runner = PipelineRunner(context)
        try {
            repeat(2) {
                runner.start(context.assets)
                assertEquals(runner.uiState.value.errorMessage, MonitorStatus.Monitoring, runner.uiState.value.status)
                assertNull("Full pipeline model availability", runner.uiState.value.warningMessage)
                repeat(46) { runner.onAudioChunk(FloatArray(512)) }
                val scored = PipelineRunner::class.java.getDeclaredField("lastScamExplain").apply { isAccessible = true }
                val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
                while ((scored.get(runner) as String).isEmpty() && System.nanoTime() < deadline) Thread.sleep(10)
                assertFalse("Asynchronous native scoring did not complete", (scored.get(runner) as String).isEmpty())
                runner.stop()
                assertEquals(MonitorStatus.Idle, runner.uiState.value.status)
                assertFalse(runner.uiState.value.isInitializing)
            }
        } finally { runner.release() }
    }

    @Test fun stopWaitsForNativeReadersInsteadOfUsingATimedRelease() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val runner = PipelineRunner(context)
        val field = PipelineRunner::class.java.getDeclaredField("lifecycleLock").apply { isAccessible = true }
        val lock = field.get(runner) as ReentrantReadWriteLock
        val entered = CountDownLatch(1)
        val finishInference = CountDownLatch(1)
        val stopped = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(2)
        try {
            val inference = executor.submit {
                lock.readLock().lock()
                try {
                    entered.countDown()
                    check(finishInference.await(10, TimeUnit.SECONDS))
                } finally { lock.readLock().unlock() }
            }
            assertTrue(entered.await(5, TimeUnit.SECONDS))
            val shutdown = executor.submit {
                runner.stop()
                stopped.countDown()
            }
            // Shutdown must wait for in-flight native inference.
            assertFalse("Shutdown released native state during inference", stopped.await(750, TimeUnit.MILLISECONDS))
            finishInference.countDown()
            inference.get(5, TimeUnit.SECONDS)
            shutdown.get(5, TimeUnit.SECONDS)
            assertEquals(MonitorStatus.Idle, runner.uiState.value.status)
        } finally {
            finishInference.countDown()
            executor.shutdownNow()
            runner.release()
        }
    }

    @Test fun stoppingPreservesCaptureFailureForTheUi() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val runner = PipelineRunner(context)
        try {
            runner.reportCaptureError("Microphone unavailable")
            runner.stop()
            assertEquals(MonitorStatus.Idle, runner.uiState.value.status)
            assertEquals("Microphone unavailable", runner.uiState.value.errorMessage)
        } finally { runner.release() }
    }
}
