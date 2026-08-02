// Module-level build.gradle.kts (app/)

// ---------------------------------------------------------------------------
// Auto-download sherpa-onnx AAR if missing (e.g. fresh clone on a new machine)
// This runs transparently before any compilation - no manual steps needed.
// ---------------------------------------------------------------------------
val sherpaAarFile = file("libs/sherpa-onnx-1.13.4.aar")
if (!sherpaAarFile.exists()) {
    println("sherpa-onnx AAR not found. Downloading (~37 MB)...")
    sherpaAarFile.parentFile.mkdirs()
    val url = java.net.URL(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-1.13.4.aar"
    )
    url.openStream().use { input ->
        sherpaAarFile.outputStream().use { output ->
            input.copyTo(output)
        }
    }
    println("sherpa-onnx AAR downloaded successfully.")
}
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.echoguard"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.echoguard"
        minSdk = 26      // AASIST-L/Zipformer inference is CPU-heavy; low-end
                         // devices below this are unlikely to hit acceptable
                         // latency anyway, per the Day-1 test results
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-hackathon"
    }

    buildTypes {
        release {
            isMinifyEnabled = false  // keep off for the hackathon build - simpler
                                     // debugging matters more than APK size here
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

    // AASIST-L's aasist_l.onnx.data companion file and other large model
    // assets shouldn't be compressed - matches the note in acoustic/README.md
    // about keeping the .onnx/.onnx.data pair together.
    androidResources {
        noCompress += listOf("onnx", "bin", "data")
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += setOf("**/libonnxruntime.so", "**/libc++_shared.so")
        }
    }
}

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

    // JSON parsing for exemplar_embeddings.json (semantic/android/ScamClassifier.kt)
    implementation("org.json:json:20240303")

    // Microsoft ONNX Runtime (MUST be listed before sherpa-onnx so pickFirst selects this library)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    // sherpa-onnx local AAR
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))

    // whisper.cpp is temporarily disabled for the hackathon demo
    // to avoid the complex NDK compilation step.
}
