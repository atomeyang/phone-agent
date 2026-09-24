package com.example.gemma4d8400.neuropilot;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Extracts the packaged runtime (models, runner, native libraries) into
 * /data/local/tmp/gemma4-d8400 and reports which files are missing.
 *
 * <p>Android's package manager compresses or blocks some of these extensions,
 * so the Gradle build keeps them stored (`noCompress`) and this class copies
 * them byte for byte, preserving the executable bit on the runner and scripts.
 */
public final class RuntimeInstaller {

    private static final String TAG = "Gemma4Runtime";
    private static final String[] COMMON_REQUIRED_FILES = {
            "libMNN.so",
            "libneuron_runtime.so.9.3.1",
            "libneuron_adapter.so.9.3.1",
            "config.json",
            "llm_config.json",
            "llm.mnn",
            "llm.mnn.weight",
            "ple_embeddings_int4.bin",
            "tokenizer.mtok"
    };
    /**
     * Files whose size must match the packaged asset even when the version
     * marker is present: the marker only changes when the build metadata does,
     * and a rebuilt runner (or config) has to reach the device on an upgrade.
     */
    private static final String[] SIZE_CHECKED_FILES = {
            BuildConfig.RUNNER_NAME, BuildConfig.BRIDGE_NAME, "agent-config.json", "libMNN.so"
    };

    public interface Progress {
        void onProgress(String message);
    }

    private final AssetManager assets;
    private final File root;
    private final Progress progress;
    private boolean installedSomething;

    public RuntimeInstaller(AssetManager assets, File root, Progress progress) {
        this.assets = assets;
        this.root = root;
        this.progress = progress;
    }

    /** @return the list of problems found (empty when the runtime is usable). */
    public List<String> ensureInstalled() {
        final List<String> problems = new ArrayList<>();
        try {
            if (!runtimeVersionMatches() || !runtimeFilesCurrent()) {
                installPackagedRuntime();
                installedSomething = true;
            }
        } catch (Exception error) {
            Log.e(TAG, "runtime extraction failed", error);
            problems.add("Runtime extraction failed: " + error.getMessage());
            return problems;
        }
        for (final String relative : requiredFiles()) {
            if (!new File(root, relative).isFile()) {
                problems.add(relative);
            }
        }
        return problems;
    }

    /** True when this run replaced files, i.e. the resident service must restart. */
    public boolean didInstall() {
        return installedSomething;
    }

    private List<String> requiredFiles() {
        final List<String> files = new ArrayList<>();
        files.add(BuildConfig.RUNNER_NAME);
        files.add(BuildConfig.BRIDGE_NAME);
        files.add("agent-config.json");
        for (final String relative : COMMON_REQUIRED_FILES) {
            files.add(relative);
        }
        for (final String relative : BuildConfig.VISION_MODEL_FILES.split(";")) {
            if (!relative.isEmpty()) {
                files.add(relative);
            }
        }
        return files;
    }

    private boolean runtimeVersionMatches() {
        final File versionFile = new File(root, ".runtime-version");
        if (!versionFile.isFile()) {
            return false;
        }
        try (RandomAccessFile input = new RandomAccessFile(versionFile, "r")) {
            final byte[] value = new byte[(int) input.length()];
            input.readFully(value);
            return BuildConfig.RUNTIME_ASSET_VERSION.equals(
                    new String(value, StandardCharsets.UTF_8));
        } catch (IOException ignored) {
            return false;
        }
    }

    private boolean runtimeFilesCurrent() {
        for (final String relative : SIZE_CHECKED_FILES) {
            final File destination = new File(root, relative);
            if (!destination.isFile()) {
                return false;
            }
            final long packaged = packagedAssetSize("runtime/" + relative);
            if (packaged >= 0 && destination.length() != packaged) {
                Log.i(TAG, "runtime file out of date: " + relative + " (device "
                        + destination.length() + " != packaged " + packaged + ")");
                return false;
            }
        }
        return true;
    }

    private void installPackagedRuntime() throws Exception {
        final List<String> packagedFiles = new ArrayList<>();
        collectPackagedFiles("runtime", "", packagedFiles);
        int index = 0;
        for (final String relative : packagedFiles) {
            final String assetPath = "runtime/" + relative;
            final File destination = new File(root, relative);
            final long packagedSize = packagedAssetSize(assetPath);
            index++;
            if (destination.isFile() && packagedSize >= 0
                    && destination.length() == packagedSize) {
                continue;
            }
            if (progress != null) {
                progress.onProgress("Installing runtime " + index + "/" + packagedFiles.size()
                        + ": " + relative);
            }
            copyPackagedFile(assetPath, destination);
        }
        writeRuntimeVersion();
    }

    private void writeRuntimeVersion() throws IOException {
        final File versionFile = new File(root, ".runtime-version");
        try (FileOutputStream output = new FileOutputStream(versionFile)) {
            output.write(BuildConfig.RUNTIME_ASSET_VERSION.getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        versionFile.setReadable(true, false);
        versionFile.setWritable(true, false);
    }

    private void collectPackagedFiles(String assetPath, String relativePath, List<String> output)
            throws IOException {
        final String[] children = assets.list(assetPath);
        if (children == null || children.length == 0) {
            if (!relativePath.isEmpty()) {
                output.add(relativePath);
            }
            return;
        }
        for (final String child : children) {
            final String childAsset = assetPath + "/" + child;
            final String childRelative =
                    relativePath.isEmpty() ? child : relativePath + "/" + child;
            collectPackagedFiles(childAsset, childRelative, output);
        }
    }

    private long packagedAssetSize(String assetPath) {
        try (AssetFileDescriptor descriptor = assets.openFd(assetPath)) {
            return descriptor.getLength();
        } catch (IOException ignored) {
            try (InputStream input = assets.open(assetPath)) {
                return input.available();
            } catch (IOException unavailable) {
                return -1;
            }
        }
    }

    private void copyPackagedFile(String assetPath, File destination) throws Exception {
        final File parent = destination.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("Cannot create " + parent);
        }
        final File temporary = new File(destination.getPath() + ".part");
        try (InputStream input = assets.open(assetPath);
                FileOutputStream output = new FileOutputStream(temporary)) {
            final byte[] buffer = new byte[8 * 1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            output.getFD().sync();
        }
        if (destination.exists() && !destination.delete()) {
            throw new IOException("Cannot replace " + destination);
        }
        if (!temporary.renameTo(destination)) {
            throw new IOException("Cannot publish " + destination);
        }
        destination.setReadable(true, false);
        destination.setWritable(true, false);
        if (destination.getName().equals(BuildConfig.RUNNER_NAME)
                || destination.getName().endsWith(".sh")) {
            destination.setExecutable(true, false);
        }
    }
}
