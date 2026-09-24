package com.example.gemma4d8400.neuropilot;

import android.os.SystemClock;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * Reads CPU/GPU/DDR current frequencies through sysfs/procfs so the host process
 * can correlate clock state with TTFT and decode rate.
 *
 * <p>The probes are best-effort: when a sysfs node is absent (which is common on
 * emulator builds), the returned map simply omits the corresponding key.</p>
 */
final class PerfMonitor {
    static final String LOG_TAG = "PhoneVLM-Perf";

    private static final String CPUFREQ_ROOT = "/sys/devices/system/cpu/cpufreq";
    private static final String DEVFREQ_ROOT = "/sys/class/devfreq";

    private final boolean screenOn;
    private final String scenario;
    private long firstSampleAt;
    private long lastSampleAt;
    private int sampleCount;

    PerfMonitor(boolean screenOn, String scenario) {
        this.screenOn = screenOn;
        this.scenario = scenario;
    }

    void begin() {
        firstSampleAt = SystemClock.elapsedRealtime();
        lastSampleAt = firstSampleAt;
        sampleCount = 0;
        Log.i(LOG_TAG, String.format(Locale.US,
                "[SCENARIO] %s screen=%s begin", scenario, screenOn ? "ON" : "OFF"));
        logSample("begin");
    }

    /**
     * Sample the current frequencies and append them to the perf log.
     * The returned snapshot is the structured payload so callers can
     * reason about CPU/GPU/DDR rates in tests.
     */
    Snapshot sample(String phase) {
        Map<String, Long> cpu = readCpuFreqKHz();
        Map<String, Long> gpu = readGpuFreqKHz();
        Map<String, Long> ddr = readDdrFreqKHz();
        long now = SystemClock.elapsedRealtime();
        long deltaMs = now - lastSampleAt;
        lastSampleAt = now;
        sampleCount++;
        Snapshot snapshot = new Snapshot(now, phase, screenOn, scenario, cpu, gpu, ddr, deltaMs);
        logSample("phase=" + phase + " delta_ms=" + deltaMs);
        return snapshot;
    }

    void finish() {
        Snapshot snap = sample("finish");
        Log.i(LOG_TAG, String.format(Locale.US,
                "[SCENARIO] %s screen=%s samples=%d duration_ms=%d avg_cpu_mhz=%.1f",
                scenario, screenOn ? "ON" : "OFF", sampleCount,
                snap.now - firstSampleAt, snap.averageCpuMHz()));
    }

    private void logSample(String label) {
        Map<String, Long> cpu = readCpuFreqKHz();
        Map<String, Long> gpu = readGpuFreqKHz();
        Map<String, Long> ddr = readDdrFreqKHz();
        Log.i(LOG_TAG, String.format(Locale.US,
                "[%s] %s cpu=%s gpu=%s ddr=%s",
                scenario, label,
                format(cpu), format(gpu), format(ddr)));
    }

    static Map<String, Long> readCpuFreqKHz() {
        // Per-core scaling_cur_freq (most reliable on MTK).
        Map<Integer, Long> perCore = new TreeMap<>();
        File root = new File("/sys/devices/system/cpu");
        File[] entries = root.listFiles();
        if (entries != null) {
            for (File cpu : entries) {
                String name = cpu.getName();
                if (!name.startsWith("cpu") || !name.substring(3).matches("\\d+")) {
                    continue;
                }
                int index = Integer.parseInt(name.substring(3));
                long kHz = readFirstLong(new File(cpu, "cpufreq/scaling_cur_freq"));
                if (kHz > 0) {
                    perCore.put(index, kHz);
                }
            }
        }
        if (perCore.isEmpty()) {
            // Some kernels only expose policy0
            File policy = new File(CPUFREQ_ROOT, "policy0/scaling_cur_freq");
            long kHz = readFirstLong(policy);
            if (kHz > 0) {
                return Collections.singletonMap("policy0", kHz);
            }
            return Collections.emptyMap();
        }
        Map<String, Long> result = new LinkedHashMap<>();
        long total = 0;
        for (Map.Entry<Integer, Long> entry : perCore.entrySet()) {
            result.put("cpu" + entry.getKey(), entry.getValue());
            total += entry.getValue();
        }
        result.put("cpu_avg", total / perCore.size());
        return result;
    }

    static Map<String, Long> readGpuFreqKHz() {
        Map<String, Long> result = new LinkedHashMap<>();
        File devfreq = new File(DEVFREQ_ROOT);
        File[] devices = devfreq.listFiles();
        if (devices != null) {
            for (File device : devices) {
                String name = device.getName().toLowerCase(Locale.US);
                if (!name.contains("gpu") && !name.contains("mali") && !name.contains("kgsl")) {
                    continue;
                }
                long kHz = readFirstLong(new File(device, "cur_freq"));
                if (kHz <= 0) {
                    continue;
                }
                result.put(name, kHz);
            }
        }
        // Fallbacks for Mediatek-specific paths.
        for (String path : new String[] {
                "/proc/gpufreq/gpufreq_cur_freq",
                "/proc/mali/utilization",
                "/sys/kernel/debug/mali/utilization"
        }) {
            long kHz = readFirstLong(new File(path));
            if (kHz > 0) {
                result.put("proc:" + new File(path).getName(), kHz);
            }
        }
        return result;
    }

    static Map<String, Long> readDdrFreqKHz() {
        Map<String, Long> result = new LinkedHashMap<>();
        File devfreq = new File(DEVFREQ_ROOT);
        File[] devices = devfreq.listFiles();
        if (devices != null) {
            for (File device : devices) {
                String name = device.getName().toLowerCase(Locale.US);
                if (!name.contains("ddr") && !name.contains("mem") && !name.contains("dram")) {
                    continue;
                }
                long kHz = readFirstLong(new File(device, "cur_freq"));
                if (kHz <= 0) {
                    continue;
                }
                result.put(name, kHz);
            }
        }
        return result;
    }

    private static long readFirstLong(File file) {
        if (!file.canRead()) {
            return -1;
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line = reader.readLine();
            if (line == null) {
                return -1;
            }
            line = line.trim();
            return Long.parseLong(line);
        } catch (IOException | NumberFormatException ignored) {
            return -1;
        }
    }

    private static String format(Map<String, Long> values) {
        if (values.isEmpty()) {
            return "{}";
        }
        StringBuilder builder = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Long> entry : values.entrySet()) {
            if (!first) {
                builder.append(", ");
            }
            first = false;
            builder.append(entry.getKey()).append('=').append(entry.getValue() / 1000).append("MHz");
        }
        return builder.append('}').toString();
    }

    /** Aggregated frequency sample exposed to the rest of the app. */
    static final class Snapshot {
        final long now;
        final String phase;
        final boolean screenOn;
        final String scenario;
        final Map<String, Long> cpuKHz;
        final Map<String, Long> gpuKHz;
        final Map<String, Long> ddrKHz;
        final long deltaMs;

        Snapshot(long now, String phase, boolean screenOn, String scenario,
                Map<String, Long> cpuKHz, Map<String, Long> gpuKHz,
                Map<String, Long> ddrKHz, long deltaMs) {
            this.now = now;
            this.phase = phase;
            this.screenOn = screenOn;
            this.scenario = scenario;
            this.cpuKHz = cpuKHz;
            this.gpuKHz = gpuKHz;
            this.ddrKHz = ddrKHz;
            this.deltaMs = deltaMs;
        }

        double averageCpuMHz() {
            Long avg = cpuKHz.get("cpu_avg");
            return avg == null ? 0.0 : avg / 1000.0;
        }

        double peakGpuMHz() {
            long peak = 0;
            for (Long value : gpuKHz.values()) {
                if (value != null && value > peak) {
                    peak = value;
                }
            }
            return peak / 1000.0;
        }

        double peakDdrMHz() {
            long peak = 0;
            for (Long value : ddrKHz.values()) {
                if (value != null && value > peak) {
                    peak = value;
                }
            }
            return peak / 1000.0;
        }

        @Override
        public String toString() {
            return String.format(Locale.US,
                    "Snapshot{scenario=%s, screen=%s, phase=%s, delta_ms=%d, cpu_avg_mhz=%.1f, gpu_peak_mhz=%.1f, ddr_peak_mhz=%.1f}",
                    scenario, screenOn ? "ON" : "OFF", phase, deltaMs,
                    averageCpuMHz(), peakGpuMHz(), peakDdrMHz());
        }
    }
}