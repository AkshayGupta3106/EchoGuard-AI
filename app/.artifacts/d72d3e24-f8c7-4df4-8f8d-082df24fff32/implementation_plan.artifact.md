# Fix Unresolved Reference for `OfflineNemoCtcModelConfig`

The project is failing to build because `OfflineNemoCtcModelConfig` is not found in the current version of `sherpa-onnx` (1.13.4). Research shows that it has been renamed to `OfflineNemoEncDecCtcModelConfig` and the corresponding parameter in `OfflineModelConfig` is `nemo`.

## Proposed Changes

### Semantic Stream

#### [MODIFY] [IndicConformerLiveTranscriber.kt](file:///D:/Download/EchoGuard-AI/app/app/src/main/java/com.echoguard/semantic/IndicConformerLiveTranscriber.kt)

- Update the import from `OfflineNemoCtcModelConfig` to `OfflineNemoEncDecCtcModelConfig`.
- Update the `init` function to use `OfflineNemoEncDecCtcModelConfig` and the `nemo` parameter when creating `OfflineModelConfig`.

## Verification Plan

### Automated Tests
- Run `./gradlew :app:compileDebugKotlin` to ensure the compilation error is resolved.
