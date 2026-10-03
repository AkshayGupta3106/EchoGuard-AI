import java.io.FileOutputStream
import java.io.InputStream
import java.net.URI
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Fetch the native ASR dependency when absent.
val sherpaAarFile = file("libs/sherpa-onnx-1.13.4.aar")
if (!sherpaAarFile.exists()) {
    println("sherpa-onnx AAR not found. Downloading (~37 MB)...")
    sherpaAarFile.parentFile.mkdirs()

    val sherpaUrl = URI(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-1.13.4.aar"
    ).toURL()

    sherpaUrl.openStream().use { input: InputStream ->
        sherpaAarFile.outputStream().use { output: FileOutputStream ->
            input.copyTo(output)
        }
    }

    println("sherpa-onnx AAR downloaded successfully.")
}

// Use Microsoft's ONNX Runtime 1.27.0/JNI pair with sherpa 1.13.4.
// Exclude the duplicate runtime from a generated sherpa AAR, not the source AAR.
val filteredSherpaAar = layout.buildDirectory.file("dependencies/sherpa-onnx-1.13.4-java-runtime.aar")
val prepareSherpaAar by tasks.registering {
    inputs.file(sherpaAarFile)
    outputs.file(filteredSherpaAar)
    doLast {
        val output = filteredSherpaAar.get().asFile
        output.parentFile.mkdirs()
        ZipFile(sherpaAarFile).use { source ->
            ZipOutputStream(output.outputStream()).use { target ->
                source.entries().asSequence().forEach { entry ->
                    if (!entry.name.endsWith("/libonnxruntime.so")) {
                        target.putNextEntry(ZipEntry(entry.name).apply { time = entry.time })
                        if (!entry.isDirectory) source.getInputStream(entry).use { it.copyTo(target) }
                        target.closeEntry()
                    }
                }
            }
        }
    }
}

// Stage VAD bytes into generated assets; a missing source is a build error.
val vadSource = rootProject.file("../assets/acoustic/silero_vad.onnx")
val generatedVadAssets = layout.buildDirectory.dir("generated/modelAssets")
val stageVadAsset by tasks.registering(Copy::class) {
    inputs.file(vadSource)
    from(vadSource)
    into(generatedVadAssets.map { it.dir("models") })
    doFirst { check(vadSource.isFile) { "Missing VAD asset: $vadSource" } }
}

android {
    namespace = "com.echoguard"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.echoguard"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-hackathon"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    // Keep model assets uncompressed for native runtime access.
    androidResources {
        noCompress += listOf("onnx", "bin", "data")
    }
    sourceSets.getByName("main").assets.srcDir(generatedVadAssets)

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += setOf("**/libc++_shared.so")
        }
    }
}

tasks.named("preBuild") { dependsOn(stageVadAsset) }

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.4")
    implementation("androidx.activity:activity-compose:1.9.1")

    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // JSON parsing for bundled semantic exemplar embeddings.
    implementation("org.json:json:20240303")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")

    implementation(files(filteredSherpaAar).builtBy(prepareSherpaAar))
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"),
        "exclude" to listOf("sherpa-onnx-1.13.4.aar"))))
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")

}
