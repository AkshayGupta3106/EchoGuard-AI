package com.echoguard.acoustic

import android.content.Context
import java.io.File

/** Cache packaged model bytes and any bundled external weights. */
internal fun copyModelAsset(context: Context, assetPath: String, cacheName: String): File {
    val model = File(context.cacheDir, cacheName)
    context.assets.open(assetPath).use { input -> model.outputStream().use { input.copyTo(it) } }
    val companionPath = "$assetPath.data"
    val directory = assetPath.substringBeforeLast('/', "")
    val companion = File(context.cacheDir, "$cacheName.data")
    val hasCompanion = context.assets.list(directory).orEmpty()
        .contains(companionPath.substringAfterLast('/'))
    if (hasCompanion) {
        context.assets.open(companionPath).use { input -> companion.outputStream().use { input.copyTo(it) } }
    } else {
        companion.delete()
    }
    return model
}
