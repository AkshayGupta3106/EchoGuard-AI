package com.echoguard.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.activity.viewModels
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PhoneInTalk
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.echoguard.fusion.Action
import com.echoguard.pipeline.AppLanguage
import com.echoguard.pipeline.CallLog
import com.echoguard.pipeline.MonitorStatus
import com.echoguard.pipeline.TimelineUiEntry
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ──────────────────────────────────────────────────────────────────────────────
// Light/dark design tokens.
// ──────────────────────────────────────────────────────────────────────────────

val LocalThemeIsDark = compositionLocalOf { false }

private val BgBeige: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFF121212) else Color(0xFFF8F6F0)
private val SurfaceWhite: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFF1E1E1E) else Color(0xFFFFFFFF)
private val DividerColor: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFF333333) else Color(0xFFE2E0D8)

private val TextDark: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFFF5F5F5) else Color(0xFF1E1E1E)
private val TextMuted: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFFBBBBBB) else Color(0xFF65635D)
private val TextMutedLight: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFFAAAAAA) else Color(0xFF716D63)

private val AccentRed: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFFE57373) else Color(0xFFB83A35)
private val AccentGreen: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFF81C784) else Color(0xFF155A38)
private val AccentAmber: Color @Composable get() = if (LocalThemeIsDark.current) Color(0xFFFFB74D) else Color(0xFF9A5400)

// ──────────────────────────────────────────────────────────────────────────────
// Activity
// ──────────────────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {

    private val pipelineViewModel: PipelineViewModel by viewModels()

    private var pendingAction: (() -> Unit)? = null

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        if (result.values.all { it }) {
            pendingAction?.invoke()
        } else {
            android.widget.Toast.makeText(this, "Microphone permission is needed for audio analysis. Enable it in Settings and try again.", android.widget.Toast.LENGTH_LONG).show()
        }
        pendingAction = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val viewModel: PipelineViewModel = viewModel()
            val themeMode by viewModel.themeMode.collectAsState()
            val isSystemDark = androidx.compose.foundation.isSystemInDarkTheme()
            val isDark = when (themeMode) {
                ThemeMode.LIGHT -> false
                ThemeMode.DARK -> true
                ThemeMode.SYSTEM -> isSystemDark
            }
            SideEffect {
                val style = SystemBarStyle.auto(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT) { isDark }
                val navigationStyle = SystemBarStyle.auto(android.graphics.Color.TRANSPARENT, android.graphics.Color.rgb(18, 18, 18)) { isDark }
                enableEdgeToEdge(statusBarStyle = style, navigationBarStyle = navigationStyle)
            }
            CompositionLocalProvider(LocalThemeIsDark provides isDark) {
                EchoGuardTheme {
                    EchoGuardScreen(onRequestPermissions = ::requestPermissionsAndCheck, viewModel = viewModel)
                }
            }
        }
    }

    override fun onStop() {
        // Foreground-only capture: save the session and release the microphone.
        pipelineViewModel.stopDemo()
        super.onStop()
    }

    private fun requestPermissionsAndCheck(onGranted: () -> Unit) {
        val perms = mutableListOf(
            Manifest.permission.RECORD_AUDIO,
        )
        val needed = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isEmpty()) {
            onGranted()
            return
        }
        pendingAction = onGranted
        requestPermissionLauncher.launch(needed.toTypedArray())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Theme
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun EchoGuardTheme(content: @Composable () -> Unit) {
    val scheme = echoGuardColorScheme(LocalThemeIsDark.current)
    MaterialTheme(colorScheme = scheme, content = content)
}

internal fun echoGuardColorScheme(isDark: Boolean): ColorScheme = if (isDark) darkColorScheme(
    background = Color(0xFF121212), surface = Color(0xFF1E1E1E),
    primary = Color(0xFFF5F5F5), onPrimary = Color(0xFF121212),
    secondary = Color(0xFF81C784), onSecondary = Color(0xFF121212),
    error = Color(0xFFE57373), onError = Color(0xFF121212),
    onBackground = Color(0xFFF5F5F5), onSurface = Color(0xFFF5F5F5),
    surfaceVariant = Color(0xFF2B2B2B), onSurfaceVariant = Color(0xFFBBBBBB), outline = Color(0xFF777777),
) else lightColorScheme(
    background = Color(0xFFF8F6F0), surface = Color.White,
    primary = Color(0xFF1E1E1E), onPrimary = Color.White,
    secondary = Color(0xFF155A38), onSecondary = Color.White,
    error = Color(0xFFB83A35), onError = Color.White,
    onBackground = Color(0xFF1E1E1E), onSurface = Color(0xFF1E1E1E),
    surfaceVariant = Color(0xFFF0EFEB), onSurfaceVariant = Color(0xFF65635D), outline = Color(0xFF716D63),
)

// ──────────────────────────────────────────────────────────────────────────────
// Root screen - Scaffold with Bottom Nav
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun EchoGuardScreen(
    onRequestPermissions: (onGranted: () -> Unit) -> Unit,
    viewModel: PipelineViewModel = viewModel(),
) {
    var currentTab by remember { mutableIntStateOf(0) }
    
    Scaffold(
        containerColor = BgBeige,
        bottomBar = {
            Column(modifier = Modifier.navigationBarsPadding().padding(bottom = 12.dp)) {
                HorizontalDivider(color = DividerColor, thickness = 1.dp)
                NavigationBar(
                    containerColor = BgBeige,
                    tonalElevation = 0.dp,
                    modifier = Modifier.height(60.dp)
                ) {
                    val tabs = listOf("HOME", "MICROPHONE", "HISTORY")
                    tabs.forEachIndexed { index, title ->
                        NavigationBarItem(
                            icon = { }, // Text-only bottom nav
                            label = { 
                                Text(
                                    title, 
                                    fontSize = 9.sp, 
                                    fontWeight = if (currentTab == index) FontWeight.Bold else FontWeight.Medium,
                                    letterSpacing = 1.sp,
                                    color = if (currentTab == index) TextDark else TextMutedLight
                                ) 
                            },
                            selected = currentTab == index,
                            onClick = { currentTab = index },
                            colors = NavigationBarItemDefaults.colors(
                                indicatorColor = Color.Transparent
                            )
                        )
                    }
                }
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding).fillMaxSize()) {
            when (currentTab) {
                0 -> HomeTab(viewModel, onRequestPermissions, onNavigateToHistory = { currentTab = 2 })
                1 -> LiveTab(viewModel, onRequestPermissions)
                2 -> HistoryTab(viewModel)
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Home tab: microphone controls, risk summary, and recent history.
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun HomeTab(
    viewModel: PipelineViewModel, 
    onRequestPermissions: (onGranted: () -> Unit) -> Unit,
    onNavigateToHistory: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val history by viewModel.history.collectAsState()
    val switching by viewModel.isSwitching.collectAsState()
    
    val isActive = uiState.status != MonitorStatus.Idle && !uiState.isInitializing
    val risk = uiState.currentRiskPercent

    val totalScreened = history.size
    val totalScams = history.count { it.action == Action.BLOCK }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp)
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(TextMutedLight))
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    "ECHOGUARD",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp,
                    color = TextMuted
                )
            }

            val themeMode by viewModel.themeMode.collectAsState()
            val themeText = when (themeMode) { 
                ThemeMode.LIGHT -> "LIGHT" 
                ThemeMode.DARK -> "DARK" 
                ThemeMode.SYSTEM -> "SYSTEM" 
            }
            Text(
                text = themeText,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = TextMuted,
                letterSpacing = 1.sp,
                modifier = Modifier
                    .clip(RoundedCornerShape(4.dp))
                    .clickable {
                        val next = when (themeMode) {
                            ThemeMode.SYSTEM -> ThemeMode.LIGHT
                            ThemeMode.LIGHT -> ThemeMode.DARK
                            ThemeMode.DARK -> ThemeMode.SYSTEM
                        }
                        viewModel.setThemeMode(next)
                    }
                    .padding(8.dp)
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            if (isActive) "ANALYSIS ACTIVE" else "ANALYSIS OFF",
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.5.sp,
            color = TextMutedLight
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "Listening,\nquietly.",
            fontFamily = FontFamily.Serif,
            fontSize = 48.sp,
            lineHeight = 52.sp,
            fontWeight = FontWeight.Normal,
            color = TextDark
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            "Analyze speech for cloned voices and scam intent.",
            fontSize = 13.sp,
            color = TextMuted,
            lineHeight = 20.sp,
            modifier = Modifier.padding(end = 32.dp)
        )

        Spacer(modifier = Modifier.height(40.dp))
        HorizontalDivider(color = DividerColor)
        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth().padding(end = 48.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            StatItem(totalScreened.toString(), "SCREENED", TextDark)
            StatItem(totalScams.toString(), "SCAMS", AccentRed)
        }

        Spacer(modifier = Modifier.height(16.dp))
        HorizontalDivider(color = DividerColor)
        Spacer(modifier = Modifier.height(24.dp))

        // Microphone controls and current risk.
        PipelineDiagnostics(viewModel)
        Spacer(modifier = Modifier.height(16.dp))
        if (isActive) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(4.dp))
                    .background(AccentRed)
                    .clickable { viewModel.stopDemo() }
                    .padding(20.dp)
            ) {
                Column {
                    Text(
                        "MICROPHONE ACTIVE — TAP TO STOP",
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        color = MaterialTheme.colorScheme.onError
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Microphone Analysis - risk $risk",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.onError
                        )
                        Icon(
                            Icons.Default.PhoneInTalk,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onError,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
        } else {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(4.dp))
                    .background(SurfaceWhite)
                    .border(1.dp, DividerColor, RoundedCornerShape(4.dp))
                    .clickable {
                        if (!switching) onRequestPermissions { viewModel.startDemo() }
                    }
                    .padding(20.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    if (uiState.isInitializing) "INITIALIZING MODELS..." else "START MICROPHONE",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp,
                    color = if (uiState.isInitializing) TextMuted else TextDark
                )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Recent session previews.
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "RECENT",
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp,
                color = TextMutedLight
            )
            Text(
                "All calls ↗",
                fontSize = 11.sp,
                color = TextDark,
                modifier = Modifier.clickable { onNavigateToHistory() }
            )
        }
        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider(color = DividerColor)
        
        if (history.isEmpty()) {
            Spacer(modifier = Modifier.height(32.dp))
            Text(
                "No recent calls yet.",
                fontSize = 13.sp,
                color = TextMuted,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center
            )
        } else {
            history.take(3).forEach { log ->
                val timeStr = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault()).format(Date(log.timestamp))
                val statusStr = when (log.action) {
                    Action.BLOCK -> "High risk - Stop recommended"
                    Action.WARN -> "Suspicious - Warned"
                    Action.MONITOR -> "No warning"
                }
                val riskColor = when (log.action) {
                    Action.BLOCK -> AccentRed
                    Action.WARN -> AccentAmber
                    Action.MONITOR -> TextDark
                }
                RecentCallRow(
                    title = log.title, 
                    subtitle = "$statusStr - $timeStr", 
                    risk = log.riskScorePercent.toString(), 
                    riskColor = riskColor
                )
                HorizontalDivider(color = DividerColor)
            }
        }

        Spacer(modifier = Modifier.weight(1f))
        
        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
fun PipelineDiagnostics(viewModel: PipelineViewModel) {
    val state by viewModel.uiState.collectAsState()
    Column(modifier = Modifier.fillMaxWidth()) {
        state.errorMessage?.let { Text(it, fontSize = 12.sp, color = AccentRed, modifier = Modifier.padding(top = 8.dp)) }
        state.warningMessage?.let { Text(it, fontSize = 12.sp, color = AccentAmber, modifier = Modifier.padding(top = 8.dp)) }
    }
}

@Composable
fun StatItem(value: String, label: String, color: Color) {
    Column {
        Text(
            value,
            fontSize = 28.sp,
            fontFamily = FontFamily.Serif,
            color = color
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            label,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.sp,
            color = TextMuted
        )
    }
}

@Composable
fun RecentCallRow(title: String, subtitle: String, risk: String, riskColor: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                title,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                color = TextDark
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                subtitle,
                fontSize = 11.sp,
                color = TextMuted
            )
        }
        Text(
            risk,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            color = riskColor
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// History Tab
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun HistoryTab(viewModel: PipelineViewModel) {
    val history by viewModel.history.collectAsState()
    
    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 24.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "SESSION HISTORY",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = TextDark,
                letterSpacing = 1.sp
            )
            if (history.isNotEmpty()) {
                Text(
                    "Clear All",
                    fontSize = 12.sp,
                    color = AccentRed,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.clickable { viewModel.callHistoryManager.clearAll() }
                )
            }
        }
        
        HorizontalDivider(color = DividerColor)
        
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 24.dp)
        ) {
            if (history.isEmpty()) {
                item {
                    Box(modifier = Modifier.fillMaxWidth().padding(48.dp), contentAlignment = Alignment.Center) {
                        Text("No microphone sessions yet.", color = TextMuted)
                    }
                }
            } else {
                items(history, key = { it.id }) { log ->
                    HistoryItem(
                        log = log, 
                        onDelete = { viewModel.callHistoryManager.deleteLog(log.id) }
                    )
                    HorizontalDivider(color = DividerColor)
                }
            }
        }
    }
}

@Composable
fun HistoryItem(log: CallLog, onDelete: () -> Unit) {
    var showTranscript by rememberSaveable(log.id) { mutableStateOf(false) }
    val timeStr = SimpleDateFormat("MMM dd, yyyy • HH:mm:ss", Locale.getDefault()).format(Date(log.timestamp))
    val statusStr = when (log.action) {
        Action.BLOCK -> "Stop recommended"
        Action.WARN -> "Warned"
        Action.MONITOR -> "No warning"
    }
    val riskColor = when (log.action) {
        Action.BLOCK -> AccentRed
        Action.WARN -> AccentAmber
        Action.MONITOR -> AccentGreen
    }
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(log.title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = TextDark)
                Spacer(modifier = Modifier.width(8.dp))
                Box(modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(riskColor.copy(alpha = 0.1f)).padding(horizontal = 6.dp, vertical = 2.dp)) {
                    Text("Risk ${log.riskScorePercent}", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = riskColor)
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text("$statusStr · $timeStr", fontSize = 12.sp, color = TextMuted)
            if (log.transcriptSnippet.isNotBlank()) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    log.transcriptSnippet + if (log.transcript == null || log.transcript.length > log.transcriptSnippet.length) "…" else "",
                    fontSize = 13.sp,
                    color = TextMutedLight,
                    fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
                )
            }
            if (!log.transcript.isNullOrBlank() || log.transcriptSnippet.isNotBlank()) {
                TextButton(onClick = { showTranscript = true }) {
                    Text(if (log.transcript != null) "VIEW FULL TRANSCRIPT" else "VIEW SAVED SNIPPET")
                }
            }
        }
        IconButton(onClick = onDelete) {
            Icon(Icons.Default.Delete, contentDescription = "Delete", tint = TextMutedLight)
        }
    }
    if (showTranscript) HistoryTranscriptDialog(log) { showTranscript = false }
}

@Composable
fun HistoryTranscriptDialog(log: CallLog, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(log.title) },
        text = {
            Column(modifier = Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState())) {
                Text(if (log.transcript == null) "SAVED SNIPPET" else "FULL TRANSCRIPT", fontWeight = FontWeight.Bold)
                if (log.transcript == null) {
                    Spacer(Modifier.height(12.dp))
                    Text("Only a preview was saved for this entry. The full transcript is unavailable.")
                }
                Spacer(Modifier.height(16.dp))
                SelectionContainer {
                    Text(log.transcript ?: log.transcriptSnippet, color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 15.sp, lineHeight = 24.sp)
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("CLOSE") } },
        containerColor = MaterialTheme.colorScheme.surface,
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Live analysis tab: transcript, risk, and timeline.
// ──────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LiveTab(viewModel: PipelineViewModel, onRequestPermissions: (onGranted: () -> Unit) -> Unit) {
    val uiState by viewModel.uiState.collectAsState()
    val switching by viewModel.isSwitching.collectAsState()
    val isHindi = uiState.uiLanguage == AppLanguage.HINDI
    val isActive = uiState.status != MonitorStatus.Idle && !uiState.isInitializing
    val listState = rememberLazyListState()

    LaunchedEffect(uiState.timeline.size) {
        if (uiState.timeline.isNotEmpty()) listState.animateScrollToItem(0)
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "MICROPHONE ANALYSIS",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = TextDark,
                letterSpacing = 1.sp
            )
            LanguageToggle(
                selected = uiState.uiLanguage,
                enabled = !isActive && !uiState.isInitializing && !switching,
                onLanguageSelected = { viewModel.setLanguage(it) }
            )
        }

        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(bottom = 32.dp),
        ) {
            item {
                RiskBanner(
                    riskPercent = uiState.currentRiskPercent,
                    action = uiState.currentAction,
                    isHindi = isHindi,
                )
            }

            item {
                PipelineDiagnostics(viewModel)
            }

            item {
                if (isActive) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(SurfaceWhite)
                            .border(1.dp, AccentRed, RoundedCornerShape(8.dp))
                            .clickable { viewModel.stopDemo() }
                            .padding(16.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("STOP MICROPHONE", color = AccentRed, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(TextDark)
                            .clickable { if (!switching) onRequestPermissions { viewModel.startDemo() } }
                            .padding(16.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            if (uiState.isInitializing) "INITIALIZING..." else "START MICROPHONE",
                            color = SurfaceWhite, 
                            fontWeight = FontWeight.Bold, 
                            fontSize = 13.sp
                        )
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
                        if (isHindi) "टाइमलाइन" else "TIMELINE",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextMuted,
                        letterSpacing = 1.sp,
                        modifier = Modifier.padding(bottom = 4.dp, start = 4.dp)
                    )
                }
                items(uiState.timeline.reversed()) { entry ->
                    TimelineRow(entry)
                }
            }

            if (!isActive && uiState.timeline.isEmpty()) {
                item { InfoCard(isHindi) }
            }
        }
    }
}

@Composable
fun RiskBanner(riskPercent: Int, action: Action, isHindi: Boolean) {
    val accent = when (action) {
        Action.MONITOR -> AccentGreen
        Action.WARN -> AccentAmber
        Action.BLOCK -> AccentRed
    }
    
    val statusTextEn = when (action) {
        Action.MONITOR -> "NO WARNING"
        Action.WARN -> "SUSPICIOUS"
        Action.BLOCK -> "FRAUD DETECTED"
    }
    val statusTextHi = when (action) {
        Action.MONITOR -> "कोई चेतावनी नहीं"
        Action.WARN -> "संदिग्ध"
        Action.BLOCK -> "धोखाधड़ी"
    }

    val animatedRisk by animateIntAsState(targetValue = riskPercent, animationSpec = tween(900), label = "")

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(SurfaceWhite)
            .border(1.dp, DividerColor, RoundedCornerShape(12.dp))
            .padding(24.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top
        ) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(accent))
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
                    color = TextMuted,
                    fontWeight = FontWeight.Medium
                )
            }
            
            Text(
                "$animatedRisk%",
                fontSize = 56.sp,
                fontFamily = FontFamily.Serif,
                color = TextDark,
                modifier = Modifier.offset(y = (-8).dp)
            )
        }
    }
}

@Composable
fun LiveTranscriptCard(transcript: String, isHindi: Boolean) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(SurfaceWhite)
            .border(1.dp, DividerColor, RoundedCornerShape(12.dp))
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
            text = transcript.ifEmpty { "..." },
            fontSize = 15.sp,
            color = if (transcript.isEmpty()) TextMuted else TextDark,
            lineHeight = 24.sp,
            minLines = 3,
        )
    }
}

@Composable
fun TimelineRow(entry: TimelineUiEntry) {
    val dotColor = when {
        entry.riskScorePercent >= 65 -> AccentRed
        entry.riskScorePercent >= 35 -> AccentAmber
        else                         -> AccentGreen
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp, horizontal = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(entry.time, fontSize = 13.sp, color = TextMuted, modifier = Modifier.width(48.dp))
        Box(modifier = Modifier.padding(top = 6.dp).size(6.dp).clip(CircleShape).background(dotColor))
        Spacer(Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(entry.text, fontSize = 14.sp, color = TextDark, lineHeight = 20.sp)
            Spacer(Modifier.height(6.dp))
            Text("${entry.riskScorePercent}% Risk • ${entry.action.name.lowercase()}", fontSize = 12.sp, color = TextMuted)
        }
    }
}

@Composable
fun InfoCard(isHindi: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(SurfaceWhite)
            .border(1.dp, DividerColor, RoundedCornerShape(8.dp))
            .padding(16.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Icon(Icons.Default.Info, contentDescription = null, tint = TextMuted, modifier = Modifier.size(16.dp).padding(top = 2.dp))
        Spacer(Modifier.width(12.dp))
        Text(
            if (isHindi) "माइक्रोफ़ोन शुरू करें और बोलें।" else "Start the microphone and speak.",
            fontSize = 13.sp, color = TextMuted, lineHeight = 20.sp
        )
    }
}

@Composable
fun LanguageToggle(selected: AppLanguage, enabled: Boolean = true, onLanguageSelected: (AppLanguage) -> Unit) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(2.dp),
        horizontalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        listOf(AppLanguage.ENGLISH to "EN", AppLanguage.HINDI to "HI").forEach { (lang, label) ->
            val isSelected = selected == lang
            Surface(
                onClick = { if (enabled) onLanguageSelected(lang) },
                shape = RoundedCornerShape(4.dp),
                color = if (isSelected) SurfaceWhite else Color.Transparent,
                modifier = Modifier.clip(RoundedCornerShape(4.dp)),
            ) {
                Text(
                    label,
                    fontSize = 11.sp,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                    color = if (isSelected) TextDark else TextMuted,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                )
            }
        }
    }
}
