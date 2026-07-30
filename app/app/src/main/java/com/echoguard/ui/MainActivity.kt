package com.echoguard.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.MicOff
import androidx.compose.material.icons.outlined.Security
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.echoguard.fusion.Action
import com.echoguard.pipeline.AppLanguage
import com.echoguard.pipeline.MonitorStatus
import com.echoguard.pipeline.TimelineUiEntry

// ──────────────────────────────────────────────────────────────────────────────
// Design tokens - Linear / Vercel style minimalism
// ──────────────────────────────────────────────────────────────────────────────

private val BgBlack       = Color(0xFF000000)
private val SurfaceCard   = Color(0xFF111111) // Stark dark grey
private val CardBorder    = Color(0xFF27272A) // Zinc-800

private val EmeraldSafe   = Color(0xFF10B981) // Flat Emerald
private val AmberWarn     = Color(0xFFF59E0B) // Flat Amber
private val RoseDanger    = Color(0xFFE11D48) // Flat Rose
private val PureWhite     = Color(0xFFFFFFFF)

private val TextPrimary   = Color(0xFFFFFFFF)
private val TextSecond    = Color(0xFFA1A1AA) // Zinc-400
private val TextMuted     = Color(0xFF71717A) // Zinc-500

// ──────────────────────────────────────────────────────────────────────────────
// Activity
// ──────────────────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { /* permissions result handled in requestPermissionsAndCheck */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val runtime = Runtime.getRuntime()
                val usedMem = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
                val maxMem = runtime.maxMemory() / 1024 / 1024
                val memInfo = "Memory: ${usedMem}MB / ${maxMem}MB"

                val trace = android.util.Log.getStackTraceString(throwable)
                android.util.Log.e("EchoGuard", "Uncaught exception on ${thread.name}\n$memInfo", throwable)
                val file = java.io.File(getExternalFilesDir(null), "crash_log.txt")
                java.io.FileWriter(file, true).use {
                    it.append("CRASH AT ${java.util.Date()} on ${thread.name}:\n")
                    it.append("$memInfo\n")
                    it.append("$trace\n\n")
                    it.flush()
                }
            } catch (_: Throwable) {}
            android.os.Process.killProcess(android.os.Process.myPid())
        }
        setContent {
            EchoGuardTheme {
                EchoGuardScreen(onRequestPermissions = ::requestPermissionsAndCheck)
            }
        }
    }

    private fun requestPermissionsAndCheck(): Boolean {
        val perms = mutableListOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE,
        )
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            perms.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val needed = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isEmpty()) return true
        requestPermissionLauncher.launch(needed.toTypedArray())
        return false
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Theme
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun EchoGuardTheme(content: @Composable () -> Unit) {
    val scheme = darkColorScheme(
        background        = BgBlack,
        surface           = SurfaceCard,
        primary           = PureWhite,
        secondary         = EmeraldSafe,
        error             = RoseDanger,
        onBackground      = TextPrimary,
        onSurface         = TextPrimary,
        surfaceVariant    = Color(0xFF18181B), // Zinc-900
        onSurfaceVariant  = TextSecond,
        outline           = CardBorder,
    )
    MaterialTheme(colorScheme = scheme, content = content)
}

// ──────────────────────────────────────────────────────────────────────────────
// Root screen
// ──────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EchoGuardScreen(
    onRequestPermissions: () -> Boolean,
    viewModel: PipelineViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    val isHindi = uiState.uiLanguage == AppLanguage.HINDI
    val isActive = uiState.status != MonitorStatus.Idle && !uiState.isInitializing
    val listState = rememberLazyListState()

    // Auto-scroll timeline to latest entry
    LaunchedEffect(uiState.timeline.size) {
        if (uiState.timeline.isNotEmpty()) listState.animateScrollToItem(0)
    }

    Scaffold(
        containerColor = BgBlack,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "EchoGuard",
                        fontWeight = FontWeight.Medium,
                        fontSize = 18.sp,
                        color = TextPrimary,
                        letterSpacing = (-0.5).sp
                    )
                },
                actions = {
                    LanguageToggle(
                        selected = uiState.uiLanguage,
                        enabled = !isActive && !uiState.isInitializing,
                        onLanguageSelected = { viewModel.setLanguage(it) }
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = BgBlack,
                ),
            )
        }
    ) { padding ->
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(bottom = 32.dp),
        ) {
            item {
                Spacer(Modifier.height(4.dp))
                RiskBanner(
                    riskPercent = uiState.currentRiskPercent,
                    action = uiState.currentAction,
                    isHindi = isHindi,
                )
            }

            item {
                AnimatedContent(
                    targetState = when {
                        uiState.isInitializing         -> "loading"
                        uiState.status == MonitorStatus.Idle -> "idle"
                        else                           -> "active"
                    },
                    transitionSpec = {
                        fadeIn(tween(300)) togetherWith fadeOut(tween(200))
                    },
                    label = "controlState"
                ) { state ->
                    when (state) {
                        "idle"    -> StartButton(isHindi) {
                            if (onRequestPermissions()) viewModel.startDemo()
                        }
                        "loading" -> LoadingCard(isHindi)
                        else      -> StopButton(isHindi) { viewModel.stopDemo() }
                    }
                }
            }

            if (isActive) {
                item {
                    LiveTranscriptCard(uiState.liveTranscript, isHindi)
                }
            }

            if (uiState.timeline.isNotEmpty()) {
                item {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        if (isHindi) "TIMELINE" else "TIMELINE",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextMuted,
                        letterSpacing = 1.sp,
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                }
                items(uiState.timeline.reversed()) { entry ->
                    TimelineRow(entry, isHindi)
                }
            }

            if (!isActive && uiState.timeline.isEmpty()) {
                item { InfoCard(isHindi) }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Risk Banner — Typography driven
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun RiskBanner(
    riskPercent: Int,
    action: Action,
    isHindi: Boolean,
) {
    val accent = when (action) {
        Action.MONITOR -> EmeraldSafe
        Action.WARN -> AmberWarn
        Action.BLOCK -> RoseDanger
    }
    
    val statusTextEn = when (action) {
        Action.MONITOR -> "PROTECTED"
        Action.WARN -> "SUSPICIOUS"
        Action.BLOCK -> "FRAUD DETECTED"
    }
    val statusTextHi = when (action) {
        Action.MONITOR -> "सुरक्षित"
        Action.WARN -> "संदिग्ध"
        Action.BLOCK -> "धोखाधड़ी"
    }

    val animatedRisk by animateIntAsState(
        targetValue = riskPercent,
        animationSpec = tween(900, easing = FastOutSlowInEasing),
        label = "riskAnim"
    )

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(SurfaceCard)
            .border(1.dp, CardBorder, RoundedCornerShape(12.dp))
            .padding(24.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top
        ) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(accent)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        if (isHindi) statusTextHi else statusTextEn,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = accent,
                        letterSpacing = 1.sp
                    )
                }
                Spacer(modifier = Modifier.height(32.dp))
                Text(
                    if (isHindi) "जोखिम स्कोर" else "Risk Score",
                    fontSize = 14.sp,
                    color = TextSecond,
                    fontWeight = FontWeight.Medium
                )
            }
            
            Text(
                "$animatedRisk%",
                fontSize = 56.sp,
                fontWeight = FontWeight.Light,
                color = TextPrimary,
                letterSpacing = (-2).sp,
                modifier = Modifier.offset(y = (-8).dp)
            )
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Control buttons - Flat, stark contrast
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun StartButton(isHindi: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth().height(56.dp),
        shape = RoundedCornerShape(8.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = PureWhite,
            contentColor = BgBlack,
        ),
        contentPadding = PaddingValues(0.dp),
    ) {
        Text(
            if (isHindi) "सुरक्षा शुरू करें" else "Start Monitoring",
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium,
            letterSpacing = 0.sp,
        )
    }
}

@Composable
fun StopButton(isHindi: Boolean, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth().height(56.dp),
        shape = RoundedCornerShape(8.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, CardBorder),
        colors = ButtonDefaults.outlinedButtonColors(
            contentColor = TextPrimary,
            containerColor = BgBlack
        ),
    ) {
        Text(
            if (isHindi) "सुरक्षा रोकें" else "Stop Monitoring",
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium,
        )
    }
}

@Composable
fun LoadingCard(isHindi: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(SurfaceCard)
            .border(1.dp, CardBorder, RoundedCornerShape(8.dp))
            .padding(20.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(20.dp),
            color = TextPrimary,
            strokeWidth = 2.dp,
        )
        Text(
            if (isHindi) "मॉडल लोड हो रहे हैं..." else "Initializing models...",
            fontWeight = FontWeight.Medium,
            color = TextPrimary,
            fontSize = 14.sp,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Live transcript - Editorial layout
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun LiveTranscriptCard(transcript: String, isHindi: Boolean) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(SurfaceCard)
            .border(1.dp, CardBorder, RoundedCornerShape(12.dp))
            .padding(20.dp)
    ) {
        Text(
            if (isHindi) "लाइव ट्रांसक्रिप्ट" else "LIVE TRANSCRIPT",
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            color = TextMuted,
            letterSpacing = 1.sp,
        )
        Spacer(Modifier.height(16.dp))
        Text(
            text = transcript.ifEmpty {
                if (isHindi) "..." else "..."
            },
            fontSize = 15.sp,
            color = if (transcript.isEmpty()) TextSecond else TextPrimary,
            lineHeight = 24.sp,
            minLines = 3,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Timeline row - Flat transaction style
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun TimelineRow(entry: TimelineUiEntry, isHindi: Boolean) {
    val dotColor = when {
        entry.riskScorePercent >= 65 -> RoseDanger
        entry.riskScorePercent >= 35 -> AmberWarn
        else                         -> EmeraldSafe
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp, horizontal = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        // Simple time column
        Text(
            entry.time,
            fontSize = 13.sp,
            color = TextMuted,
            modifier = Modifier.width(48.dp)
        )
        
        // Timeline dot
        Box(
            modifier = Modifier
                .padding(top = 6.dp)
                .size(6.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        
        Spacer(Modifier.width(16.dp))
        
        // Content
        Column(modifier = Modifier.weight(1f)) {
            Text(
                entry.text,
                fontSize = 14.sp,
                color = TextPrimary,
                lineHeight = 20.sp,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "${entry.riskScorePercent}% ${if (isHindi) "जोखिम" else "Risk"} • ${entry.action.name.lowercase()}",
                fontSize = 12.sp,
                color = TextSecond,
            )
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Info card
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun InfoCard(isHindi: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .border(1.dp, CardBorder, RoundedCornerShape(8.dp))
            .padding(16.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Icon(
            Icons.Default.Info,
            contentDescription = null,
            tint = TextMuted,
            modifier = Modifier.size(16.dp).padding(top = 2.dp),
        )
        Spacer(Modifier.width(12.dp))
        Text(
            if (isHindi)
                "डेमो सीधे माइक का उपयोग करता है। असली कॉल के लिए स्पीकर मोड चालू करें।"
            else
                "Demo uses mic directly. For real calls, switch to Speaker mode so both sides of the call are captured.",
            fontSize = 13.sp,
            color = TextSecond,
            lineHeight = 20.sp,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Language toggle - Minimalist pill
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun LanguageToggle(
    selected: AppLanguage,
    enabled: Boolean = true,
    onLanguageSelected: (AppLanguage) -> Unit,
) {
    Row(
        modifier = Modifier
            .padding(end = 8.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(SurfaceCard)
            .border(1.dp, CardBorder, RoundedCornerShape(6.dp))
            .padding(2.dp),
        horizontalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        listOf(AppLanguage.ENGLISH to "EN", AppLanguage.HINDI to "HI").forEach { (lang, label) ->
            val isSelected = selected == lang
            Surface(
                onClick = { if (enabled) onLanguageSelected(lang) },
                shape = RoundedCornerShape(4.dp),
                color = if (isSelected) Color(0xFF27272A) else Color.Transparent,
                modifier = Modifier.clip(RoundedCornerShape(4.dp)),
            ) {
                Text(
                    label,
                    fontSize = 11.sp,
                    fontWeight = if (isSelected) FontWeight.Medium else FontWeight.Normal,
                    color = if (isSelected) TextPrimary else if (enabled) TextSecond else TextMuted,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                )
            }
        }
    }
}
