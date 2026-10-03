package com.echoguard.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.core.view.WindowCompat
import androidx.lifecycle.ViewModelProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** UI-only checks. No microphone capture, recordings, or phone calls. */
@RunWith(AndroidJUnit4::class)
class ThemeAndPermissionsTest {
    private fun contrast(a: Color, b: Color): Float =
        (maxOf(a.luminance(), b.luminance()) + .05f) / (minOf(a.luminance(), b.luminance()) + .05f)

    @Test fun bothPalettesHaveReadableTextAndButtons() {
        for (dark in listOf(false, true)) {
            val scheme = echoGuardColorScheme(dark)
            for ((foreground, background) in listOf(
                scheme.onPrimary to scheme.primary, scheme.onSecondary to scheme.secondary,
                scheme.onError to scheme.error, scheme.onSurface to scheme.surface,
                scheme.onBackground to scheme.background, scheme.onSurfaceVariant to scheme.surfaceVariant,
            )) assertTrue("Insufficient contrast in dark=$dark", contrast(foreground, background) >= 4.5f)
        }
    }

    @Test fun themeSwitchUpdatesStatusAndNavigationIcons() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val activity = instrumentation.startActivitySync(Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
        lateinit var vm: PipelineViewModel
        instrumentation.runOnMainSync { vm = ViewModelProvider(activity)[PipelineViewModel::class.java] }
        val previous = vm.themeMode.value
        try {
            for (mode in listOf(ThemeMode.LIGHT, ThemeMode.DARK, ThemeMode.LIGHT)) {
                instrumentation.runOnMainSync { vm.setThemeMode(mode) }
                val deadline = System.nanoTime() + 5_000_000_000L
                var matches = false
                while (!matches && System.nanoTime() < deadline) {
                    instrumentation.waitForIdleSync()
                    instrumentation.runOnMainSync {
                        val controller = WindowCompat.getInsetsController(activity.window, activity.window.decorView)
                        matches = controller.isAppearanceLightStatusBars == (mode == ThemeMode.LIGHT) &&
                            controller.isAppearanceLightNavigationBars == (mode == ThemeMode.LIGHT && android.os.Build.VERSION.SDK_INT >= 27)
                    }
                    if (!matches) Thread.sleep(20)
                }
                assertTrue("System icons do not follow $mode", matches)
            }
        } finally { instrumentation.runOnMainSync { vm.setThemeMode(previous); activity.finish() } }
    }

    @Test fun appDoesNotDeclarePhoneMonitoringServiceOrPermissions() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        @Suppress("DEPRECATION")
        val info = context.packageManager.getPackageInfo(context.packageName, PackageManager.GET_PERMISSIONS or PackageManager.GET_SERVICES)
        val permissions = info.requestedPermissions?.toList().orEmpty()
        assertTrue(Manifest.permission.RECORD_AUDIO in permissions)
        for (permission in listOf(Manifest.permission.READ_PHONE_STATE, Manifest.permission.POST_NOTIFICATIONS,
            Manifest.permission.FOREGROUND_SERVICE, Manifest.permission.FOREGROUND_SERVICE_MICROPHONE)) {
            assertFalse(permission, permission in permissions)
        }
        assertFalse(info.services.orEmpty().any { it.name.endsWith("CallMonitorService") })
    }
}
